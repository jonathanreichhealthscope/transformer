#pragma once
#include <cuda_runtime.h>

// Declare CUDA kernels
__global__ void layer_norm_stats_kernel(
    const float* input,
    float* mean,
    float* variance,
    const int hidden_size,
    const int batch_size
);

__global__ void layer_norm_kernel(
    const float* input,
    const float* mean,
    const float* variance,
    const float* gamma,
    const float* beta,
    float* output,
    const int hidden_size,
    const int batch_size,
    const float eps
);