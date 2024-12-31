#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "feed_forward_kernels.cuh"
#include <stdio.h>

namespace cuda_kernels {
    // Helper struct declaration only
    template<typename... Args>
    struct KernelLauncher {
        template<typename KernelFunc>
        static void launch(KernelFunc kernel,
                         dim3 grid_dim,
                         dim3 block_dim,
                         size_t shared_mem,
                         cudaStream_t stream,
                         Args... args);
    };

    // Template declarations
    template<typename... Args>
    void launch_cuda_kernel(void(*kernel)(Args...),
                          dim3 grid_dim,
                          dim3 block_dim,
                          size_t shared_mem,
                          cudaStream_t stream,
                          Args... args);

    // Extern template declarations
    extern template void launch_cuda_kernel(
        void(*)(const float*, const float*, float*, int, int, int),
        dim3, dim3, size_t, cudaStream_t,
        const float*, const float*, float*, int, int, int);

    extern template void launch_cuda_kernel(
        void(*)(const float*, float*, int),
        dim3, dim3, size_t, cudaStream_t,
        const float*, float*, int);
}

// Global cuBLAS handle
extern cublasHandle_t cublas_handle;

// CUDA initialization/cleanup
void initialize_cuda();
void cleanup_cuda();