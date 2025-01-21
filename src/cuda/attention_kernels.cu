#include "../include/cuda/attention_kernels.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void attention_scores_kernel(const float* Q, const float* K, float* scores,
                                        const float scale, int seq_len, int head_dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < seq_len && col < seq_len) {
        float sum = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            sum += Q[row * head_dim + i] * K[col * head_dim + i];
        }
        scores[row * seq_len + col] = sum * scale;
    }
}

__global__ void softmax_kernel(float* scores, int seq_len) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < seq_len) {
        // Find max for numerical stability
        float max_val = scores[row * seq_len];
        for (int i = 1; i < seq_len; i++) {
            max_val = max(max_val, scores[row * seq_len + i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            scores[row * seq_len + i] = expf(scores[row * seq_len + i] - max_val);
            sum += scores[row * seq_len + i];
        }

        // Normalize
        for (int i = 0; i < seq_len; i++) {
            scores[row * seq_len + i] /= sum;
        }
    }
}