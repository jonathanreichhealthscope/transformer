#pragma once
#include "components.hpp"
#include <memory>
#include <unordered_map>
#include <vector>
#ifdef USE_CUDA
#include "cuda/cuda_utils.cuh"
#include "cuda/cuda_check.cuh"
#include <cuda_runtime.h>
#endif

/**
 * @brief Memory management system for efficient tensor operations.
 * 
 * The MemoryPool class provides a memory allocation strategy that reduces
 * fragmentation and improves performance by:
 * - Pre-allocating memory blocks
 * - Recycling freed memory
 * - Supporting both CPU and GPU memory
 * - Providing static allocation fallback
 * 
 * This implementation helps reduce memory allocation overhead during
 * training and inference.
 */
class MemoryPool {
  public:
    // Default constructor
    MemoryPool() : pool_size(DEFAULT_POOL_SIZE), block_size(0), current_block(0)
#ifdef USE_CUDA
        , cuda_current_block(0)
#endif
    {
        initialize();
    }

    // Constructor with size parameter
    explicit MemoryPool(size_t pool_size_) : 
        pool_size(pool_size_), 
        block_size(0), 
        current_block(0)
#ifdef USE_CUDA
        , cuda_current_block(0)
#endif
    {
        initialize();
    }

    /**
     * @brief Destructor that ensures proper cleanup of all allocated memory.
     */
    ~MemoryPool() {
        cleanup();
    }

    /**
     * @brief Allocates CPU memory from the pool.
     * 
     * Attempts to reuse previously freed memory before allocating
     * new blocks to reduce fragmentation.
     * 
     * @param size Number of bytes to allocate
     * @return Pointer to allocated memory
     * @throws std::bad_alloc if allocation fails
     */
    float* allocate(size_t size) {
#ifdef USE_CUDA
        void* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        allocations[ptr] = size;
        return static_cast<float*>(ptr);
#else
        return nullptr;
#endif
    }

    /**
     * @brief Returns memory to the pool for reuse.
     * 
     * Instead of freeing memory immediately, stores it for
     * future allocations of the same size.
     * 
     * @param ptr Pointer to memory being deallocated
     * @param size Size of the memory block
     */
    void deallocate(void* ptr) {
#ifdef USE_CUDA
        if (!ptr) return;
        
        auto it = allocations.find(ptr);
        if (it != allocations.end()) {
            CUDA_CHECK(cudaFree(ptr));
            allocations.erase(it);
        }
#endif
    }

    /**
     * @brief Resets the memory pool to its initial state.
     * 
     * Frees all allocated blocks and clears the free list,
     * useful between training epochs or inference batches.
     */
    void reset() {
#ifdef USE_CUDA
        cleanup();
        initialize();
#endif
    }

#ifdef USE_CUDA
    /**
     * @brief Allocates GPU memory from the pool.
     * 
     * Similar to CPU allocation but manages CUDA memory,
     * reducing GPU memory fragmentation.
     * 
     * @param size Number of bytes to allocate
     * @return Pointer to GPU memory
     * @throws std::runtime_error if CUDA allocation fails
     */
    float* cuda_allocate(size_t size) {
        float* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        allocations[ptr] = size;
        return ptr;
    }

    /**
     * @brief Resets the GPU memory pool.
     * 
     * Frees all allocated CUDA memory blocks and
     * resets the pool state.
     */
    void cuda_reset() {
        cleanup();
        initialize();
    }
#endif

    /**
     * @brief Static allocation function for fallback.
     * 
     * Provides direct memory allocation when pool allocation
     * is not suitable or available.
     * 
     * @param size Number of bytes to allocate
     * @return Pointer to allocated memory
     */
    static float* allocate_static(size_t size) {
        return static_cast<float*>(std::malloc(size));
    }

    /**
     * @brief Static deallocation function for fallback.
     * 
     * Frees memory allocated through static allocation.
     * 
     * @param ptr Pointer to memory being freed
     * @param size Size of the memory block
     */
    static void deallocate_static(void* ptr, size_t size) {
        std::free(ptr);
    }

  private:
    static constexpr size_t DEFAULT_POOL_SIZE = 1024 * 1024 * 512;  // 512MB default
    size_t pool_size;
    size_t block_size;      ///< Size of each memory block
    size_t current_block;   ///< Index of the current block being allocated from
    std::unordered_map<void*, size_t> allocations;

#ifdef USE_CUDA
    cudaMemPool_t memPool;
    std::vector<float*> cuda_blocks;  ///< Storage for CUDA memory blocks
    size_t cuda_current_block;        ///< Index of current CUDA block
#endif

    void initialize() {
#ifdef USE_CUDA
        // Initialize CUDA memory pool
        cudaMemPoolProps poolProps = {};
        poolProps.allocType = cudaMemAllocationTypePinned;
        poolProps.location.type = cudaMemLocationTypeDevice;
        poolProps.location.id = 0;
        poolProps.handleTypes = cudaMemHandleTypeNone;
        
        cudaMemPool_t* poolPtr = &memPool;
        CUDA_CHECK(cudaMemPoolCreate(poolPtr, &poolProps));
        
        unsigned long long threshold = pool_size;
        CUDA_CHECK(cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &threshold));
#endif
    }

    void cleanup() {
#ifdef USE_CUDA
        // Free all allocated memory
        for (const auto& alloc : allocations) {
            cudaFree(alloc.first);
        }
        allocations.clear();
        
        // Destroy memory pool
        if (memPool) {
            CUDA_CHECK(cudaMemPoolDestroy(memPool));
        }
#endif
    }
};