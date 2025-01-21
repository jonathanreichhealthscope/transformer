#pragma once

namespace cuda_kernels {
extern "C" {
// Forward declare the kernels
__global__ void feed_forward_backward_kernel_1(const float*, const float*, float*, int, int, int);
__global__ void gelu_backward_kernel(const float*, float*, int);
__global__ void feed_forward_backward_kernel_2(const float*, const float*, float*, int, int, int);
};
} // namespace cuda_kernels