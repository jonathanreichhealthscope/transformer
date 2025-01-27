#pragma once
#include <cuda_runtime.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#define KERNEL __global__
#else
#define CUDA_CALLABLE
#define KERNEL
#endif

namespace cuda {

// Declare CUDA kernels
KERNEL void layer_norm_stats_kernel(const float* input, float* mean, float* variance,
                                  const int hidden_size, const int batch_size);

KERNEL void layer_norm_kernel(const float* input, const float* mean, const float* variance,
                            const float* gamma, const float* beta, float* output,
                            const int hidden_size, const int batch_size, const float eps);

KERNEL void layer_norm_backward_kernel(const float* grad_output, const float* input,
                                     const float* gamma, float* grad_gamma,
                                     float* grad_beta, int batch_size, int hidden_size,
                                     float eps);

} // namespace cuda