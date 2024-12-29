#include "cuda/quantization_kernels.cuh"

__global__ void convert_f32_to_f16(
    const float* input,
    __half* output,
    const int num_elements
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_elements) {
        output[tid] = __float2half(input[tid]);
    }
}

__global__ void convert_f16_to_f32(
    const __half* input,
    float* output,
    const int num_elements
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_elements) {
        output[tid] = __half2float(input[tid]);
    }
} 