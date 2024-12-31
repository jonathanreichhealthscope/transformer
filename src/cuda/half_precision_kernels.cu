#include "../../include/cuda/half_precision_kernels.cuh"

__global__ void convert_fp32_to_fp16_kernel(const float *input, __half *output,
                                            size_t size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = __float2half(input[idx]);
  }
}

__global__ void convert_fp16_to_fp32_kernel(const __half *input, float *output,
                                            size_t size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = __half2float(input[idx]);
  }
}

void launch_fp32_to_fp16(const float *input, __half *output, size_t size) {
  dim3 block(256);
  dim3 grid((size + block.x - 1) / block.x);
  convert_fp32_to_fp16_kernel<<<grid, block>>>(input, output, size);
}

void launch_fp16_to_fp32(const __half *input, float *output, size_t size) {
  dim3 block(256);
  dim3 grid((size + block.x - 1) / block.x);
  convert_fp16_to_fp32_kernel<<<grid, block>>>(input, output, size);
}