#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#define KERNEL __global__
#else
#define CUDA_CALLABLE
#define KERNEL
#endif

// Declare CUDA kernels
KERNEL void convert_f32_to_f16(
    const float* input,
    __half* output,
    const int num_elements
);

KERNEL void convert_f16_to_f32(
    const __half* input,
    float* output,
    const int num_elements
); 