#pragma once
#include "attention.hpp"
#include "matrix.hpp"

class GroupedQueryAttention {
  private:
    size_t num_heads;    // Total number of query heads
    size_t num_kv_heads; // Number of key/value heads (smaller than num_heads)
    size_t head_dim;     // Dimension of each head
    size_t hidden_size;  // Total hidden size
    float dropout_prob;

    // Projection matrices
    Matrix query_proj;  // [hidden_size, num_heads * head_dim]
    Matrix key_proj;    // [hidden_size, num_kv_heads * head_dim]
    Matrix value_proj;  // [hidden_size, num_kv_heads * head_dim]
    Matrix output_proj; // [num_heads * head_dim, hidden_size]

    // Helper methods
    Matrix repeat_kv_heads(const Matrix& kv, size_t num_repeats) const;
    Matrix compute_grouped_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                                     const AttentionMask& mask) const;

  public:
    GroupedQueryAttention(size_t hidden_size, size_t num_heads, size_t num_kv_heads,
                          size_t head_dim, float dropout_prob);

    Matrix forward(const Matrix& x, const AttentionMask& mask,
                   const std::optional<KVCache>& kv_cache = std::nullopt);

    // Getters for dimensions
    size_t get_num_heads() const {
        return num_heads;
    }
    size_t get_num_kv_heads() const {
        return num_kv_heads;
    }
    size_t get_head_dim() const {
        return head_dim;
    }
};