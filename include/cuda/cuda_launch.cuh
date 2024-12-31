#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

#ifdef __CUDACC__
// Direct kernel launch for CUDA files
#define CUDA_LAUNCH(kernel, grid, block, shared_mem, stream, ...) \
    kernel<<<grid, block, shared_mem, stream>>>(__VA_ARGS__)
#else
// Template function for non-CUDA files
template<typename KernelFunc, typename... Args>
void launchCudaKernel(
    KernelFunc kernel,
    dim3 gridDim,
    dim3 blockDim,
    size_t sharedMem,
    cudaStream_t stream,
    Args... args);

#define CUDA_LAUNCH(kernel, grid, block, shared_mem, stream, ...) \
    launchCudaKernel(kernel, grid, block, shared_mem, stream, __VA_ARGS__)
#endif 