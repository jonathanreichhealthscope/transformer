#pragma once
#include "cuda_utils.cuh"
#include <cstdio>

namespace cuda_kernels {
    template<typename KernelFunc, typename... Args>
    void launch_cuda_kernel(KernelFunc kernel, 
                          dim3 grid_dim, 
                          dim3 block_dim, 
                          size_t shared_mem, 
                          cudaStream_t stream,
                          Args... args) {
        kernel<<<grid_dim, block_dim, shared_mem, stream>>>(args...);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
    }
} 