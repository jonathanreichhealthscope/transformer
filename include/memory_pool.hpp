#pragma once
#include "components.hpp"
#include <memory>
#include <unordered_map>
#include <vector>
#ifdef USE_CUDA
#include "cuda/cuda_utils.cuh"
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
    /**
     * @brief Constructs a memory pool with specified block size.
     * @param block_size_ Size of each memory block in bytes
     */
    explicit MemoryPool(size_t block_size_);

    /**
     * @brief Destructor that ensures proper cleanup of all allocated memory.
     */
    ~MemoryPool();

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
    float* allocate(size_t size);

    /**
     * @brief Returns memory to the pool for reuse.
     * 
     * Instead of freeing memory immediately, stores it for
     * future allocations of the same size.
     * 
     * @param ptr Pointer to memory being deallocated
     * @param size Size of the memory block
     */
    void deallocate(void* ptr, size_t size);

    /**
     * @brief Resets the memory pool to its initial state.
     * 
     * Frees all allocated blocks and clears the free list,
     * useful between training epochs or inference batches.
     */
    void reset();

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
    float* cuda_allocate(size_t size);

    /**
     * @brief Resets the GPU memory pool.
     * 
     * Frees all allocated CUDA memory blocks and
     * resets the pool state.
     */
    void cuda_reset();
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
    /// Map of free blocks indexed by size
    static std::unordered_map<size_t, std::vector<void*>> free_blocks;

    /// Storage for allocated memory blocks
    std::vector<std::unique_ptr<float[]>> blocks;

    size_t block_size;      ///< Size of each memory block
    size_t current_block;   ///< Index of the current block being allocated from

#ifdef USE_CUDA
    std::vector<float*> cuda_blocks;  ///< Storage for CUDA memory blocks
    size_t cuda_current_block;        ///< Index of current CUDA block
#endif
};