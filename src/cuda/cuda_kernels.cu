#include "cuda_kernels.hpp"
#include <cuda_fp16.h>

__global__ void softmax_kernel(float* matrix, int rows, int cols) {
    int row = blockIdx.x;
    
    // Find max value
    float max_val = -INFINITY;
    for (int i = 0; i < cols; i++) {
        max_val = max(max_val, matrix[row * cols + i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        float val = exp(matrix[row * cols + i] - max_val);
        matrix[row * cols + i] = val;
        sum += val;
    }
    
    // Normalize
    for (int i = 0; i < cols; i++) {
        matrix[row * cols + i] /= sum;
    }
}

__global__ void attention_kernel(float* Q, float* K, float* V, float* output,
                               int batch_size, int seq_len, int head_dim) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = threadIdx.x;
    
    __shared__ float scores[1024];  // Assuming max sequence length
    
    // Compute attention scores
    for (int j = 0; j < seq_len; j++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += Q[b * seq_len * head_dim + i * head_dim + d] *
                    K[b * seq_len * head_dim + j * head_dim + d];
        }
        scores[j] = score / sqrt(float(head_dim));
    }
    
    // Apply softmax
    __syncthreads();
    float max_score = -INFINITY;
    for (int j = 0; j < seq_len; j++) {
        max_score = max(max_score, scores[j]);
    }
    
    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        scores[j] = exp(scores[j] - max_score);
        sum += scores[j];
    }
    
    for (int j = 0; j < seq_len; j++) {
        scores[j] /= sum;
    }
    
    // Compute weighted sum
    for (int d = 0; d < head_dim; d++) {
        float weighted_sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            weighted_sum += scores[j] * V[b * seq_len * head_dim + j * head_dim + d];
        }
        output[b * seq_len * head_dim + i * head_dim + d] = weighted_sum;
    }
} 