#pragma once
#include <cuda_runtime.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#define KERNEL __global__
#else
#define CUDA_CALLABLE
#define KERNEL
#endif

// Declare CUDA kernels
KERNEL void layer_norm_backward_kernel(const float *grad, const float *input,
                                       const float *gamma, float *dx,
                                       const int batch_size,
                                       const int hidden_size, const float eps);