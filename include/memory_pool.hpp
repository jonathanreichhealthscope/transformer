#pragma once
#include "components.hpp"
#include <vector>
#include <memory>
#ifdef USE_CUDA
#include "cuda/cuda_utils.cuh"
#endif

class MemoryPool {
private:
    std::vector<std::unique_ptr<float[]>> blocks;
    size_t block_size;
    size_t current_block;

#ifdef USE_CUDA
    std::vector<float*> cuda_blocks;
    size_t cuda_current_block;
#endif

public:
    explicit MemoryPool(size_t block_size = 1024 * 1024);
    ~MemoryPool();
    
    // CPU memory management
    float* allocate(size_t size);
    void reset();

#ifdef USE_CUDA
    // CUDA memory management
    float* cuda_allocate(size_t size);
    void cuda_reset();
#endif
}; 