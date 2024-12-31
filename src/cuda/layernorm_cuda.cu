#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/layernorm.hpp"

__global__ void layernorm_backward_kernel(const float *grad_output,
                                          const float *input,
                                          const float *gamma, float *grad_input,
                                          float *grad_gamma, float *grad_beta,
                                          int hidden_size, int batch_size,
                                          float eps) {
  extern __shared__ float shared_mem[];
  float *mean = shared_mem;
  float *variance = &shared_mem[batch_size];

  int tid = threadIdx.x;
  int bid = blockIdx.x;

  // Compute mean and variance for each sequence position
  if (tid < hidden_size) {
    float sum = 0.0f;
    float sq_sum = 0.0f;
    for (int i = 0; i < batch_size; i++) {
      float val = input[i * hidden_size + tid];
      sum += val;
      sq_sum += val * val;
    }
    mean[tid] = sum / batch_size;
    variance[tid] = sq_sum / batch_size - mean[tid] * mean[tid] + eps;
  }
  __syncthreads();

  // Compute gradients
  for (int i = tid; i < batch_size * hidden_size; i += blockDim.x) {
    int seq_pos = i / hidden_size;
    int hidden_pos = i % hidden_size;

    float x = input[i];
    float dy = grad_output[i];
    float mu = mean[hidden_pos];
    float var = variance[hidden_pos];
    float std = sqrt(var);
    float gamma_val = gamma[hidden_pos];

    // Gradient with respect to input
    float dx = gamma_val * (dy - mu) / std;
    grad_input[i] = dx;

    // Gradient with respect to gamma and beta
    atomicAdd(&grad_gamma[hidden_pos], dy * (x - mu) / std);
    atomicAdd(&grad_beta[hidden_pos], dy);
  }
}

Matrix LayerNorm::backward_cuda(const Matrix &grad_output,
                                const Matrix &input) const {
  const int batch_size = input.rows();
  const int hidden_size = input.cols();

  // Allocate device memory
  float *d_grad_output, *d_input, *d_gamma;
  float *d_grad_input, *d_grad_gamma, *d_grad_beta;

  CUDA_CHECK(cudaMalloc(&d_grad_output, grad_output.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gamma, gamma_.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_grad_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_grad_gamma, gamma_.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_grad_beta, beta_.size() * sizeof(float)));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_grad_output, grad_output.data(),
                        grad_output.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, gamma_.data(), gamma_.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Launch kernel
  int block_size = 256;
  int grid_size = (batch_size * hidden_size + block_size - 1) / block_size;
  size_t shared_mem_size =
      2 * hidden_size * sizeof(float); // For mean and variance

  layernorm_backward_kernel<<<grid_size, block_size, shared_mem_size>>>(
      d_grad_output, d_input, d_gamma, d_grad_input, d_grad_gamma, d_grad_beta,
      hidden_size, batch_size, eps_);

  // Create result matrix and copy gradients back
  Matrix grad_input(batch_size, hidden_size);

  CUDA_CHECK(cudaMemcpy(grad_input.data(), d_grad_input,
                        grad_input.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Update gamma and beta gradients
  Vector grad_gamma(hidden_size), grad_beta(hidden_size);
  CUDA_CHECK(cudaMemcpy(grad_gamma.data(), d_grad_gamma,
                        grad_gamma.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(grad_beta.data(), d_grad_beta,
                        grad_beta.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Cleanup
  CUDA_CHECK(cudaFree(d_grad_output));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_gamma));
  CUDA_CHECK(cudaFree(d_grad_input));
  CUDA_CHECK(cudaFree(d_grad_gamma));
  CUDA_CHECK(cudaFree(d_grad_beta));

  return grad_input;
}