#include "../../include/cuda/cuda_launch.cuh"

#ifndef __CUDACC__
// Implementation for non-CUDA files
template <typename KernelFunc, typename... Args>
void launchCudaKernel(KernelFunc kernel, dim3 gridDim, dim3 blockDim,
                      size_t sharedMem, cudaStream_t stream, Args... args) {
  throw std::runtime_error("CUDA kernel launch attempted in non-CUDA context");
}

// Explicit template instantiations for the specific kernel types we use
template void
launchCudaKernel<void (*)(const float *, const float *, float *, int, int, int),
                 float *, float *, float *, unsigned long, unsigned long,
                 unsigned long>(void (*)(const float *, const float *, float *,
                                         int, int, int),
                                dim3, dim3, size_t, cudaStream_t *, float *,
                                float *, float *, unsigned long, unsigned long,
                                unsigned long);

template void
launchCudaKernel<void (*)(const float *, float *, int), float *, float *,
                 unsigned long>(void (*)(const float *, float *, int), dim3,
                                dim3, size_t, cudaStream_t *, float *, float *,
                                unsigned long);
#endif