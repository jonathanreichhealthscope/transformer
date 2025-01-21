#include "../include/cuda/gqa_kernels.cuh"
#include "../include/cuda/cuda_utils.cuh"
#include <cuda_runtime.h>

__device__ void compute_attention_scores(float* scores, const float* query, const float* key,
                                      int seq_len, int head_dim, int idx) {
    float sum = 0.0f;
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    // Compute scaled dot product attention
    for (int d = 0; d < head_dim; d++) {
        sum += query[d] * key[d];
    }
    scores[idx] = sum * scale;
}

__device__ void apply_softmax(float* scores, int seq_len) {
    // Find max for numerical stability
    float max_val = scores[0];
    for (int i = 1; i < seq_len; i++) {
        max_val = max(max_val, scores[i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        scores[i] = __expf(scores[i] - max_val);
        sum += scores[i];
    }
    
    // Normalize
    for (int i = 0; i < seq_len; i++) {
        scores[i] /= sum;
    }
}

__global__ void gqa_forward_kernel(const float* query, const float* key, const float* value,
                                 float* output, const int batch_size, const int num_heads,
                                 const int num_kv_heads, const int seq_len, const int head_dim) {
    // Get thread indices
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * seq_len * num_heads;
    
    if (tid >= total_threads) return;
    
    // Calculate indices
    int batch_idx = tid / (seq_len * num_heads);
    int seq_idx = (tid / num_heads) % seq_len;
    int head_idx = tid % num_heads;
    int kv_head_idx = head_idx % num_kv_heads;  // Map query head to corresponding KV head
    
    // Calculate offsets
    int q_offset = tid * head_dim;
    int k_offset = (batch_idx * seq_len * num_kv_heads + seq_idx * num_kv_heads + kv_head_idx) * head_dim;
    int v_offset = k_offset;
    int out_offset = tid * head_dim;
    
    // Shared memory for attention scores and temporary calculations
    extern __shared__ float shared_mem[];
    float* attention_scores = shared_mem;
    
    // Compute attention scores for this query with all keys
    for (int i = 0; i < seq_len; i++) {
        compute_attention_scores(attention_scores + threadIdx.x * seq_len,
                               query + q_offset,
                               key + k_offset + i * head_dim,
                               seq_len, head_dim, i);
    }
    
    // Apply softmax to attention scores
    apply_softmax(attention_scores + threadIdx.x * seq_len, seq_len);
    
    // Compute weighted sum of values
    for (int d = 0; d < head_dim; d++) {
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            sum += attention_scores[threadIdx.x * seq_len + i] * 
                   value[v_offset + i * head_dim + d];
        }
        output[out_offset + d] = sum;
    }
}

void launch_gqa_kernel(const float* query, const float* key, const float* value,
                      float* output, const int batch_size, const int num_heads,
                      const int num_kv_heads, const int seq_len, const int head_dim,
                      cudaStream_t stream) {
    // Calculate grid and block dimensions
    int total_threads = batch_size * seq_len * num_heads;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;
    
    // Calculate shared memory size (for attention scores)
    size_t shared_mem_size = block_size * seq_len * sizeof(float);
    
    // Launch kernel
    gqa_forward_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        query, key, value, output, batch_size, num_heads, 
        num_kv_heads, seq_len, head_dim);
} 