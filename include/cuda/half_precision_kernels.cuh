#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Kernel declarations
__global__ void convert_fp32_to_fp16_kernel(const float* input, __half* output, size_t size);
__global__ void convert_fp16_to_fp32_kernel(const __half* input, float* output, size_t size);

// Helper functions that can be called from .cpp files
void launch_fp32_to_fp16(const float* input, __half* output, size_t size);
void launch_fp16_to_fp32(const __half* input, float* output, size_t size); 