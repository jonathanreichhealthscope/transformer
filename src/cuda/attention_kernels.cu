#include "../../include/cuda/cuda_utils.cuh"

__global__ void flash_attention_kernel(const float *Q, const float *K,
                                       const float *V, float *output,
                                       const int batch_size,
                                       const int seq_length,
                                       const int head_dim) {
  // Basic implementation - can be optimized further
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * seq_length * head_dim)
    return;

  const int b = idx / (seq_length * head_dim);
  const int s = (idx / head_dim) % seq_length;
  const int h = idx % head_dim;

  float sum = 0.0f;
  float max_val = -INFINITY;

  // Find max for numerical stability
  for (int i = 0; i < seq_length; ++i) {
    float qk = Q[b * seq_length * head_dim + s * head_dim + h] *
               K[b * seq_length * head_dim + i * head_dim + h];
    max_val = max(max_val, qk);
  }

  // Compute attention scores
  float denom = 0.0f;
  for (int i = 0; i < seq_length; ++i) {
    float qk = Q[b * seq_length * head_dim + s * head_dim + h] *
               K[b * seq_length * head_dim + i * head_dim + h];
    float score = exp(qk - max_val);
    sum += score * V[b * seq_length * head_dim + i * head_dim + h];
    denom += score;
  }

  output[idx] = sum / denom;
}