#pragma once
#include "components.hpp"
#include "cuda/cuda_utils.cuh"
#include "memory_pool.hpp"
#include <cuda_runtime.h>
#include <future>
#include <memory>
#include <vector>

class CudaManager {
private:
    std::unique_ptr<MemoryPool> memory_pool;

public:
    explicit CudaManager(int device_id = 0, size_t pool_size = 0) {
        CUDA_CHECK(cudaSetDevice(device_id));
        memory_pool = std::make_unique<MemoryPool>(pool_size > 0 ? pool_size : 512 * 1024 * 1024);
    }

    ~CudaManager() {
        memory_pool.reset();  // Release pool before device reset
        CUDA_CHECK(cudaDeviceReset());
    }

    void synchronize() {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void* allocate(size_t size) {
        return memory_pool->allocate(size);
    }

    void deallocate(void* ptr) {
        if (ptr) {
            memory_pool->deallocate(ptr);
        }
    }
};

class MultiGPUManager {
  private:
    std::vector<std::unique_ptr<CudaManager>> gpu_managers;
    std::vector<cudaStream_t> streams;
    int num_gpus;

    Matrix process_batch(const std::vector<Matrix>& inputs, size_t start, size_t end) {
        // Process subset of inputs on current GPU
        Matrix result(inputs[start].rows(), inputs[start].cols());
        for (size_t i = start; i < end; ++i) {
            // Add actual processing logic here
            result += inputs[i]; // Placeholder
        }
        return result;
    }

  public:
    MultiGPUManager() {
        CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
        for (int i = 0; i < num_gpus; i++) {
            gpu_managers.push_back(std::make_unique<CudaManager>(i));
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));
            streams.push_back(stream);
        }
    }

    ~MultiGPUManager() {
        for (auto& stream : streams) {
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
    }

    std::vector<Matrix> parallel_forward(const std::vector<Matrix>& inputs) {
        size_t batch_per_gpu = (inputs.size() + num_gpus - 1) / num_gpus; // Ceiling division
        std::vector<std::future<Matrix>> futures;

        for (int i = 0; i < num_gpus; i++) {
            size_t start = i * batch_per_gpu;
            size_t end = std::min(start + batch_per_gpu, inputs.size());

            if (start >= inputs.size())
                break;

            futures.push_back(std::async(std::launch::async, [this, &inputs, i, start, end]() {
                CUDA_CHECK(cudaSetDevice(i));
                return this->process_batch(inputs, start, end);
            }));
        }

        std::vector<Matrix> results;
        for (auto& f : futures) {
            results.push_back(f.get());
        }
        return results;
    }
};