#include "../include/memory_pool.hpp"
#include "../include/cuda/cuda_check.cuh"
#include <stdexcept>
#include <stdlib.h>

MemoryPool::MemoryPool(size_t block_size_)
    : block_size(block_size_), current_block(0)
#ifdef USE_CUDA
      ,
      cuda_current_block(0)
#endif
{
  // Initialize with one CPU block
  blocks.push_back(std::make_unique<float[]>(block_size));

#ifdef USE_CUDA
  // Initialize with one CUDA block
  float *cuda_ptr;
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

void MemoryPool::reset() {
  // Keep only one block and reset counter
  blocks.resize(1);
  current_block = 0;
}

#ifdef USE_CUDA
float *MemoryPool::cuda_allocate(size_t size) {
  // Check if we need a new block
  if (cuda_current_block + size > block_size) {
    // Add new CUDA block
    float *cuda_ptr;
    CUDA_CHECK(cudaMalloc(&cuda_ptr, block_size * sizeof(float)));
    cuda_blocks.push_back(cuda_ptr);
    cuda_current_block = 0;
  }

  // Allocate from current CUDA block
  float *ptr = cuda_blocks.back() + cuda_current_block;
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

std::unordered_map<size_t, std::vector<void *>> MemoryPool::free_blocks;

float *MemoryPool::allocate(size_t size) {
  // Try to reuse existing block
  auto it = free_blocks.find(size);
  if (it != free_blocks.end() && !it->second.empty()) {
    float *ptr = static_cast<float *>(it->second.back());
    it->second.pop_back();
    return ptr;
  }

// Allocate new block
#ifdef USE_CUDA
  float *ptr;
  CUDA_CHECK(
      cudaMallocHost(&ptr, size)); // Use pinned memory for better GPU transfer
  return ptr;
#else
  return static_cast<float *>(std::malloc(size)); // Standard allocation
#endif
}

void MemoryPool::deallocate(void *ptr, size_t size) {
  if (!ptr)
    return;

  // Store for reuse
  free_blocks[size].push_back(ptr);

  // Optional: Clean up if too many blocks
  if (free_blocks[size].size() > 1000) { // Arbitrary threshold
    for (size_t i = 500; i < free_blocks[size].size(); ++i) {
#ifdef USE_CUDA
      CUDA_CHECK(cudaFreeHost(free_blocks[size][i]));
#else
      std::free(free_blocks[size][i]);
#endif
    }
    free_blocks[size].resize(500);
  }
}