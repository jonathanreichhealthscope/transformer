#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void convert_f32_to_f16(const float* input, __half* output, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(input[idx]);
    }
}

__global__ void convert_f16_to_f32(const __half* input, float* output, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __half2float(input[idx]);
    }
} 