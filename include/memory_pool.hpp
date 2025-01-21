#pragma once
#include "components.hpp"
#include <memory>
#include <unordered_map>
#include <vector>
#ifdef USE_CUDA
#include "cuda/cuda_utils.cuh"
#endif

class MemoryPool {
  public:
    explicit MemoryPool(size_t block_size_);
    ~MemoryPool();

    // CPU memory management
    float* allocate(size_t size);
    void deallocate(void* ptr, size_t size);
    void reset();

#ifdef USE_CUDA
    // CUDA memory management
    float* cuda_allocate(size_t size);
    void cuda_reset();
#endif

    // Static memory pool interface
    static float* allocate_static(size_t size) {
        return static_cast<float*>(std::malloc(size));
    }
    static void deallocate_static(void* ptr, size_t size) {
        std::free(ptr);
    }

  private:
    static std::unordered_map<size_t, std::vector<void*>> free_blocks;
    std::vector<std::unique_ptr<float[]>> blocks;
    size_t block_size;
    size_t current_block;

#ifdef USE_CUDA
    std::vector<float*> cuda_blocks;
    size_t cuda_current_block;
#endif
};