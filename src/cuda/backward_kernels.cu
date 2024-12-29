#include "cuda/backward_kernels.cuh"

__global__ void
layer_norm_backward_kernel(const float *grad, const float *input,
                           const float *gamma, float *dx, const int batch_size,
                           const int hidden_size, const float eps) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  extern __shared__ float shared_mem[];
  float *mean = shared_mem;
  float *var = shared_mem + blockDim.x;
  float *sum_grad = shared_mem + 2 * blockDim.x;
  float *sum_grad_diff = shared_mem + 3 * blockDim.x;

  for (int batch_idx = tid; batch_idx < batch_size; batch_idx += stride) {
    // Compute mean and variance
    float batch_mean = 0.0f;
    float batch_var = 0.0f;

    for (int j = 0; j < hidden_size; ++j) {
      batch_mean += input[batch_idx * hidden_size + j];
    }
    batch_mean /= hidden_size;

    for (int j = 0; j < hidden_size; ++j) {
      float diff = input[batch_idx * hidden_size + j] - batch_mean;
      batch_var += diff * diff;
    }
    batch_var /= hidden_size;
    float std = sqrtf(batch_var + eps);

    // Compute gradient components
    float batch_sum_grad = 0.0f;
    float batch_sum_grad_diff = 0.0f;

    for (int j = 0; j < hidden_size; ++j) {
      float diff = input[batch_idx * hidden_size + j] - batch_mean;
      batch_sum_grad += grad[batch_idx * hidden_size + j] * gamma[j];
      batch_sum_grad_diff +=
          grad[batch_idx * hidden_size + j] * gamma[j] * diff;
    }

    // Compute final gradients
    for (int j = 0; j < hidden_size; ++j) {
      float diff = input[batch_idx * hidden_size + j] - batch_mean;
      int idx = batch_idx * hidden_size + j;
      dx[idx] = gamma[j] *
                (grad[idx] -
                 (batch_sum_grad + diff * batch_sum_grad_diff / batch_var) /
                     hidden_size) /
                std;
    }
  }
}

__global__ void feed_forward_backward_kernel_1(
    const float *grad, const float *w2, float *d_intermediate,
    const int batch_size, const int hidden_size, const int intermediate_size) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_elements = batch_size * intermediate_size;

  for (int idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
    const int batch = idx / intermediate_size;
    const int inter = idx % intermediate_size;

    float sum = 0.0f;
    for (int k = 0; k < hidden_size; ++k) {
      sum += grad[batch * hidden_size + k] * w2[inter * hidden_size + k];
    }
    d_intermediate[idx] = sum;
  }
}

__global__ void gelu_backward_kernel(float *d_intermediate, const float *input,
                                     const int size) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int idx = tid; idx < size; idx += blockDim.x * gridDim.x) {
    float x = input[idx];
    float cdf = 0.5f * (1.0f + tanhf(0.797884f * (x + 0.044715f * x * x * x)));
    float pdf = expf(-0.5f * x * x) * 0.797884f;
    d_intermediate[idx] *= (cdf + x * pdf);
  }
}

__global__ void feed_forward_backward_kernel_2(const float *d_intermediate,
                                               const float *w1, float *dx,
                                               const int batch_size,
                                               const int hidden_size,
                                               const int intermediate_size) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_elements = batch_size * hidden_size;

  for (int idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
    const int batch = idx / hidden_size;
    const int hidden = idx % hidden_size;

    float sum = 0.0f;
    for (int k = 0; k < intermediate_size; ++k) {
      sum += d_intermediate[batch * intermediate_size + k] *
             w1[hidden * intermediate_size + k];
    }
    dx[idx] = sum;
  }
}