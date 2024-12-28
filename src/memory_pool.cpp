#include "../include/memory_pool.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

// Helper function for CUDA error checking
static void check_cuda_error(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + 
                               cudaGetErrorString(error));
    }
}

float* MemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(instance().mutex);
    
    // Look for an existing block that's large enough and not in use
    for (auto& block : instance().blocks) {
        if (!block.in_use && block.size >= size) {
            block.in_use = true;
            return block.data;
        }
    }
    
    // If no suitable block found, allocate new memory
    float* data = new float[size];
    instance().blocks.push_back({data, size, true});
    return data;
}

void MemoryPool::deallocate(float* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(instance().mutex);
    
    // Find and mark the block as not in use
    auto it = std::find_if(instance().blocks.begin(), instance().blocks.end(),
                          [ptr](const Block& block) { return block.data == ptr; });
    
    if (it != instance().blocks.end()) {
        it->in_use = false;
    }
}

float* MemoryPool::cuda_allocate(size_t size) {
    float* ptr;
    check_cuda_error(
        cudaMalloc(&ptr, size * sizeof(float)),
        "CUDA memory allocation failed"
    );
    return ptr;
}

void MemoryPool::cuda_deallocate(float* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

void MemoryPool::clear_pool() {
    std::lock_guard<std::mutex> lock(instance().mutex);
    
    // Delete all CPU memory blocks
    for (const auto& block : instance().blocks) {
        delete[] block.data;
    }
    instance().blocks.clear();
}

// PooledPtr implementation
template<typename T>
PooledPtr<T>::PooledPtr(size_t size, bool cuda) : is_cuda(cuda) {
    if (cuda) {
        ptr = reinterpret_cast<T*>(MemoryPool::cuda_allocate(size));
    } else {
        ptr = reinterpret_cast<T*>(MemoryPool::allocate(size));
    }
}

template<typename T>
PooledPtr<T>::~PooledPtr() {
    if (ptr) {
        if (is_cuda) {
            MemoryPool::cuda_deallocate(reinterpret_cast<float*>(ptr));
        } else {
            MemoryPool::deallocate(reinterpret_cast<float*>(ptr));
        }
        ptr = nullptr;
    }
}

// Explicit template instantiations
template class PooledPtr<float>;
template class PooledPtr<int>; 