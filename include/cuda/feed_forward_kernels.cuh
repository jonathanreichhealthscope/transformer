#pragma once
#include "../matrix.hpp"
#include <cuda_runtime.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#define KERNEL __global__
#else
#define CUDA_CALLABLE
#define KERNEL
#endif

namespace cuda {
    void feed_forward_backward(const Matrix& grad, const Matrix& weights,
                            Matrix& dx, bool is_first_layer);

    __global__ void feed_forward_backward_kernel_1(const float* grad, const float* w2,
                                                float* d_intermediate, int batch_size,
                                                int hidden_size, int intermediate_size);

    __global__ void feed_forward_backward_kernel_2(const float* d_intermediate, const float* w1,
                                                float* dx, int batch_size,
                                                int hidden_size, int intermediate_size);
}

// Declare CUDA kernels
KERNEL void feed_forward_backward_kernel_1(const float* grad, const float* w2,
                                           float* d_intermediate, const int batch_size,
                                           const int hidden_size, const int intermediate_size);

KERNEL void gelu_backward_kernel(const float* d_intermediate, float* d_input,
                                 const int num_elements);

KERNEL void feed_forward_backward_kernel_2(const float* d_intermediate, const float* w1,
                                           float* d_dx, const int batch_size, const int hidden_size,
                                           const int intermediate_size);