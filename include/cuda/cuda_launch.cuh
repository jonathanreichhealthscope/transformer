#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <stdexcept>

#ifdef __CUDACC__
// Template function to handle CUDA kernel launches - CUDA compiler version
template<typename KernelFunc, typename... Args>
void launchCudaKernel(
    KernelFunc kernel,
    dim3 gridDim,
    dim3 blockDim,
    size_t sharedMem,
    cudaStream_t stream,
    Args... args)
{
    kernel<<<gridDim, blockDim, sharedMem, stream>>>(args...);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

// Macro that forwards to the template function
#define CUDA_LAUNCH(kernel, grid, block, shared_mem, stream, ...) \
    launchCudaKernel(kernel, grid, block, shared_mem, stream, __VA_ARGS__)

#else
// Non-CUDA compiler version - declare but don't define the triple-angle-bracket syntax
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

#endif // USE_CUDA 