#include "../include/gqa.hpp"
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
    std::cout << "=== GroupedQueryAttention::forward START ===" << std::endl;
    
    // Project input to Q, K, V
    Matrix Q = matmul(x, query_proj);
    Matrix K = matmul(x, key_proj);
    Matrix V = matmul(x, value_proj);
    
    // Handle KV cache if present
    if (kv_cache) {
        Matrix cached_K = kv_cache->key_cache;
        Matrix cached_V = kv_cache->value_cache;
        
        // Concatenate current K/V with cached K/V
        Matrix new_K(K.rows() + cached_K.rows(), K.cols());
        Matrix new_V(V.rows() + cached_V.rows(), V.cols());
        
        // Copy cached values
        for (size_t i = 0; i < cached_K.rows(); i++) {
            for (size_t j = 0; j < cached_K.cols(); j++) {
                new_K(i, j) = cached_K(i, j);
                new_V(i, j) = cached_V(i, j);
            }
        }
        
        // Copy new values
        for (size_t i = 0; i < K.rows(); i++) {
            for (size_t j = 0; j < K.cols(); j++) {
                new_K(i + cached_K.rows(), j) = K(i, j);
                new_V(i + cached_V.rows(), j) = V(i, j);
            }
        }
        
        K = std::move(new_K);
        V = std::move(new_V);
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
} 