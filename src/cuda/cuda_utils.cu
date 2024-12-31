#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/cuda_check.cuh"

namespace cuda_kernels {
    // Implementation of KernelLauncher
    template<typename... Args>
    template<typename KernelFunc>
    void KernelLauncher<Args...>::launch(
        KernelFunc kernel,
        dim3 grid_dim,
        dim3 block_dim,
        size_t shared_mem,
        cudaStream_t stream,
        Args... args) 
    {
        kernel<<<grid_dim, block_dim, shared_mem, stream>>>(args...);
        CUDA_CHECK(cudaGetLastError());
    }

    // Implementation of launch_cuda_kernel
    template<typename... Args>
    void launch_cuda_kernel(
        void(*kernel)(Args...),
        dim3 grid_dim,
        dim3 block_dim,
        size_t shared_mem,
        cudaStream_t stream,
        Args... args) 
    {
        KernelLauncher<Args...>::launch(kernel, grid_dim, block_dim, shared_mem, stream, args...);
    }

    // Explicit instantiations
    template void launch_cuda_kernel(
        void(*)(const float*, const float*, float*, int, int, int),
        dim3, dim3, size_t, cudaStream_t,
        const float*, const float*, float*, int, int, int);

    template void launch_cuda_kernel(
        void(*)(const float*, float*, int),
        dim3, dim3, size_t, cudaStream_t,
        const float*, float*, int);
} 