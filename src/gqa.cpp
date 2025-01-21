#include "../include/gqa.hpp"
#include "../include/cuda/gqa_kernels.cuh"
#include <iostream>

GroupedQueryAttention::GroupedQueryAttention(size_t hidden_size_, size_t num_heads_,
                                           size_t num_kv_heads_, size_t head_dim_,
                                           float dropout_prob_)
    : hidden_size(hidden_size_),
      num_heads(num_heads_),
      num_kv_heads(num_kv_heads_),
      head_dim(head_dim_),
      dropout_prob(dropout_prob_) {
    
    // Initialize projection matrices
    query_proj = Matrix(hidden_size, num_heads * head_dim);
    key_proj = Matrix(hidden_size, num_kv_heads * head_dim);
    value_proj = Matrix(hidden_size, num_kv_heads * head_dim);
    output_proj = Matrix(num_heads * head_dim, hidden_size);

    // Initialize weights with small random values
    query_proj.randomize(-0.02f, 0.02f);
    key_proj.randomize(-0.02f, 0.02f);
    value_proj.randomize(-0.02f, 0.02f);
    output_proj.randomize(-0.02f, 0.02f);
}

Matrix GroupedQueryAttention::repeat_kv_heads(const Matrix& kv, size_t num_repeats) const {
    // Repeat each KV head to match the number of query heads
    size_t batch_seq_len = kv.rows();
    Matrix repeated(batch_seq_len, kv.cols() * num_repeats);
    
    for (size_t i = 0; i < batch_seq_len; i++) {
        for (size_t h = 0; h < num_kv_heads; h++) {
            for (size_t d = 0; d < head_dim; d++) {
                float val = kv(i, h * head_dim + d);
                for (size_t r = 0; r < num_repeats; r++) {
                    repeated(i, (h * num_repeats + r) * head_dim + d) = val;
                }
            }
        }
    }
    return repeated;
}

Matrix GroupedQueryAttention::compute_grouped_attention(
    const Matrix& Q, const Matrix& K, const Matrix& V,
    const AttentionMask& mask) const {
    
    // Scale factor for attention scores
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // Compute attention scores
    Matrix scores = matmul(Q, K.transpose());
    scores *= scale;
    
    // Apply mask if provided
    if (!mask.mask.empty()) {
        for (size_t i = 0; i < scores.rows(); i++) {
            for (size_t j = 0; j < scores.cols(); j++) {
                if (mask.mask(i % mask.mask.rows(), j % mask.mask.cols()) == 0.0f) {
                    scores(i, j) = -std::numeric_limits<float>::infinity();
                }
            }
        }
    }
    
    // Apply softmax
    for (size_t i = 0; i < scores.rows(); i++) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < scores.cols(); j++) {
            max_val = std::max(max_val, scores(i, j));
        }
        
        float sum = 0.0f;
        for (size_t j = 0; j < scores.cols(); j++) {
            scores(i, j) = std::exp(scores(i, j) - max_val);
            sum += scores(i, j);
        }
        
        for (size_t j = 0; j < scores.cols(); j++) {
            scores(i, j) /= sum;
        }
    }
    
    // Compute attention output
    return matmul(scores, V);
}

Matrix GroupedQueryAttention::forward(const Matrix& x, const AttentionMask& mask,
                                    const std::optional<KVCache>& kv_cache) {
    std::cout << "GroupedQueryAttention::forward called with CUDA_AVAILABLE="
    #ifdef CUDA_AVAILABLE
              << "true" << std::endl;
    #else
              << "false" << std::endl;
    #endif
    
    #ifdef CUDA_AVAILABLE
        // Project input to Q, K, V first
        printf("=== GroupedQueryAttention::forward CUDA START ===\n");
        Matrix Q = matmul(x, query_proj);
        Matrix K = matmul(x, key_proj);
        Matrix V = matmul(x, value_proj);

        // Handle KV cache
        if (kv_cache) {
            K = kv_cache->key_cache;
            V = kv_cache->value_cache;
        }

        // Calculate dimensions
        size_t batch_size = x.rows();
        size_t seq_len = mask.mask.rows();

        // Move to GPU
        Matrix Q_gpu = Q.to_gpu();
        Matrix K_gpu = K.to_gpu();
        Matrix V_gpu = V.to_gpu();
        Matrix output_gpu(batch_size, hidden_size);

        // Launch CUDA kernel
        launch_gqa_kernel(
            Q_gpu.data(), K_gpu.data(), V_gpu.data(),
            output_gpu.data(), batch_size, num_heads, num_kv_heads,
            seq_len, head_dim, nullptr
        );

        return output_gpu.to_cpu();
    #else
        // CPU implementation
        std::cout << "=== GroupedQueryAttention::forward START ===" << std::endl;
        
        // Project input to Q, K, V
        Matrix Q = matmul(x, query_proj);
        Matrix K = matmul(x, key_proj);
        Matrix V = matmul(x, value_proj);
        
        // Handle KV cache if present
        if (kv_cache) {
            K = kv_cache->key_cache;
            V = kv_cache->value_cache;
        }
        
        // Repeat K/V heads to match number of query heads
        size_t num_repeats = num_heads / num_kv_heads;
        K = repeat_kv_heads(K, num_repeats);
        V = repeat_kv_heads(V, num_repeats);
        
        // Compute attention
        Matrix attention_output = compute_grouped_attention(Q, K, V, mask);
        
        // Project output
        Matrix output = matmul(attention_output, output_proj);
        
        std::cout << "=== GroupedQueryAttention::forward END ===" << std::endl;
        return output;
    #endif
} 