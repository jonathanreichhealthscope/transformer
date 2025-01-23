#pragma once
#include "cuda_utils.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "../matrix.hpp"
#include "cuda_check.cuh"

namespace cuda {
    class MemoryManager {
    public:
        static MemoryManager& instance() {
            static MemoryManager instance;
            return instance;
        }

        // Get or allocate device memory
        float* get_device_memory(size_t size) {
            try {
                if (auto it = memory_pool_.find(size); it != memory_pool_.end()) {
                    return it->second;
                }
                
                float* device_ptr;
                CUDA_CHECK(cudaMalloc(&device_ptr, size * sizeof(float)));
                memory_pool_[size] = device_ptr;
                return device_ptr;
            } catch (const std::exception& e) {
                throw std::runtime_error("CUDA memory allocation failed: " + std::string(e.what()));
            }
        }

        void clear_pool() {
            for (auto& [size, ptr] : memory_pool_) {
                cudaFree(ptr);
            }
            memory_pool_.clear();
        }

        ~MemoryManager() {
            clear_pool();
        }

    private:
        MemoryManager() = default;
        std::unordered_map<size_t, float*> memory_pool_;
    };
}

class CudaMemoryPool {
  private:
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };

    std::vector<Block> blocks;
    std::mutex mutex;
    size_t total_allocated{0};
    size_t max_allocation{1ULL << 32}; // 4GB default
    cudaStream_t stream;

  public:
    CudaMemoryPool() {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    ~CudaMemoryPool() {
        for (auto& block : blocks) {
            if (block.ptr) {
                cudaFree(block.ptr);
            }
        }
        cudaStreamDestroy(stream);
    }

    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex);

        // Try to find existing block
        for (auto& block : blocks) {
            if (!block.in_use && block.size >= size) {
                block.in_use = true;
                return block.ptr;
            }
        }

        // Allocate new block
        if (total_allocated + size > max_allocation) {
            throw std::runtime_error("CUDA memory pool limit exceeded");
        }

        void* ptr;
        CUDA_CHECK(cudaMallocAsync(&ptr, size, stream));
        blocks.push_back({ptr, size, true});
        total_allocated += size;
        return ptr;
    }

    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto& block : blocks) {
            if (block.ptr == ptr) {
                block.in_use = false;
                return;
            }
        }
    }

    void defragment() {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<Block> new_blocks;
        for (auto& block : blocks) {
            if (block.in_use) {
                new_blocks.push_back(block);
            } else {
                CUDA_CHECK(cudaFreeAsync(block.ptr, stream));
                total_allocated -= block.size;
            }
        }
        blocks = std::move(new_blocks);
    }
};

class CudaGraphManager {
  private:
    struct GraphCache {
        cudaGraph_t graph;
        cudaGraphExec_t instance;
        std::vector<void*> node_params;
        bool is_dirty;
    };

    std::unordered_map<size_t, GraphCache> graph_cache;
    cudaStream_t stream;

  public:
    CudaGraphManager() {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    template <typename F> void capture_and_execute(size_t key, F&& kernel_launch) {
        auto& cache = graph_cache[key];
        if (cache.is_dirty) {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            kernel_launch();
            cudaStreamEndCapture(stream, &cache.graph);
            cudaGraphInstantiate(&cache.instance, cache.graph, nullptr, nullptr, 0);
            cache.is_dirty = false;
        }
        cudaGraphLaunch(cache.instance, stream);
    }
};