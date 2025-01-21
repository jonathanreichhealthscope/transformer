#pragma once
#include <cuda_runtime.h>

void find_minmax_cuda(const float* input, size_t size, float* min_val, float* max_val);

__global__ void quantize_kernel(const float* input, float* output, size_t size, float scale,
                                float zero_point);

__global__ void dequantize_kernel(const float* input, float* output, size_t size, float scale,
                                  float zero_point);