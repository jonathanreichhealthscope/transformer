#include "../include/memory_pool.hpp"
#include <stdexcept>

MemoryPool::MemoryPool(size_t block_size_)
    : block_size(block_size_), current_block(0)
#ifdef USE_CUDA
    , cuda_current_block(0)
#endif
{
    // Initialize with one CPU block
    blocks.push_back(std::make_unique<float[]>(block_size));

#ifdef USE_CUDA
    // Initialize with one CUDA block
    float* cuda_ptr;
    CUDA_CHECK(cudaMalloc(&cuda_ptr, block_size * sizeof(float)));
    cuda_blocks.push_back(cuda_ptr);
#endif
}

MemoryPool::~MemoryPool() {
#ifdef USE_CUDA
    for (auto ptr : cuda_blocks) {
        cudaFree(ptr);
    }
#endif
}

float* MemoryPool::allocate(size_t size) {
    // Check if we need a new block
    if (current_block + size > block_size) {
        // Add new block if needed
        blocks.push_back(std::make_unique<float[]>(block_size));
        current_block = 0;
    }

    // Allocate from current block
    float* ptr = blocks.back().get() + current_block;
    current_block += size;
    return ptr;
}

void MemoryPool::reset() {
    // Keep only one block and reset counter
    blocks.resize(1);
    current_block = 0;
}

#ifdef USE_CUDA
float* MemoryPool::cuda_allocate(size_t size) {
    // Check if we need a new block
    if (cuda_current_block + size > block_size) {
        // Add new CUDA block
        float* cuda_ptr;
        CUDA_CHECK(cudaMalloc(&cuda_ptr, block_size * sizeof(float)));
        cuda_blocks.push_back(cuda_ptr);
        cuda_current_block = 0;
    }

    // Allocate from current CUDA block
    float* ptr = cuda_blocks.back() + cuda_current_block;
    cuda_current_block += size;
    return ptr;
}

void MemoryPool::cuda_reset() {
    // Free all but first CUDA block
    while (cuda_blocks.size() > 1) {
        cudaFree(cuda_blocks.back());
        cuda_blocks.pop_back();
    }
    cuda_current_block = 0;
}
#endif 