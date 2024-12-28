#include "cuda_manager.hpp"
#include <stdexcept>

CudaManager::CudaManager() {
    // Initialize CUDA
    cudaError_t error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to initialize CUDA device");
    }
}

CudaManager::~CudaManager() {
    cudaDeviceReset();
}

void CudaManager::synchronize() {
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA synchronization failed");
    }
}

void* CudaManager::allocate(size_t size) {
    void* ptr;
    cudaError_t error = cudaMalloc(&ptr, size);
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA memory allocation failed");
    }
    return ptr;
}

void CudaManager::deallocate(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
} 