#pragma once
#include <cuda_runtime.h>

// LayerNorm backward kernels
__global__ void layer_norm_backward_kernel(
    const float* grad,
    const float* input,
    const float* gamma,
    float* dx,
    const int batch_size,
    const int hidden_size,
    const float eps
);

// FeedForward backward kernels
__global__ void feed_forward_backward_kernel_1(
    const float* grad,
    const float* w2,
    float* d_intermediate,
    const int batch_size,
    const int hidden_size,
    const int intermediate_size
);

__global__ void gelu_backward_kernel(
    float* d_intermediate,
    const float* input,
    const int size
);

__global__ void feed_forward_backward_kernel_2(
    const float* d_intermediate,
    const float* w1,
    float* dx,
    const int batch_size,
    const int hidden_size,
    const int intermediate_size
);

// Attention backward kernels
__global__ void attention_backward_kernel(
    const float* grad,
    const float* Q,
    const float* K,
    const float* V,
    float* dQ,
    float* dK,
    float* dV,
    const int batch_size,
    const int seq_length,
    const int num_heads,
    const int head_dim
); 