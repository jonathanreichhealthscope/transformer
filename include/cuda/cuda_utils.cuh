#pragma once
#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "feed_forward_kernels.cuh"

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// cuBLAS error checking macro
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

namespace cuda_kernels {
    // CUDA kernel launch wrapper function template declaration
    template<typename KernelFunc, typename... Args>
    void launch_cuda_kernel(KernelFunc kernel, 
                           dim3 grid_dim, 
                           dim3 block_dim, 
                           size_t shared_mem, 
                           cudaStream_t stream,
                           Args... args);

    // Extern template declarations
    extern template void launch_cuda_kernel<decltype(&feed_forward_backward_kernel_1)>(
        decltype(&feed_forward_backward_kernel_1),
        dim3, dim3, size_t, cudaStream_t,
        const float*, const float*, float*, int, int, int);

    extern template void launch_cuda_kernel<decltype(&gelu_backward_kernel)>(
        decltype(&gelu_backward_kernel),
        dim3, dim3, size_t, cudaStream_t,
        const float*, float*, int);

    extern template void launch_cuda_kernel<decltype(&feed_forward_backward_kernel_2)>(
        decltype(&feed_forward_backward_kernel_2),
        dim3, dim3, size_t, cudaStream_t,
        const float*, const float*, float*, int, int, int);
}

// Macro to make kernel launches more readable
#define CUDA_LAUNCH(kernel, grid, block, shared_mem, stream, ...) \
    cuda_kernels::launch_cuda_kernel(kernel, grid, block, shared_mem, stream, __VA_ARGS__)

// Global cuBLAS handle
extern cublasHandle_t cublas_handle;

// Initialize CUDA resources
void initialize_cuda();

// Clean up CUDA resources
void cleanup_cuda();

#endif // CUDA_UTILS_CUH