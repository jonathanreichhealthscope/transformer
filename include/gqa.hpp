#pragma once
#include "attention.hpp"
#include "matrix.hpp"

/**
 * @brief Implements Grouped Query Attention for efficient transformer attention.
 * 
 * Grouped Query Attention (GQA) is an optimization that reduces memory and compute
 * requirements by using fewer key/value heads than query heads. Features include:
 * - Memory-efficient attention computation
 * - Configurable head grouping ratios
 * - KV-cache support for autoregressive generation
 * - Dropout regularization
 * 
 * Reference: https://arxiv.org/abs/2305.13245
 */
class GroupedQueryAttention {
  private:
    size_t num_heads;    ///< Total number of query heads
    size_t num_kv_heads; ///< Number of key/value heads (smaller than num_heads)
    size_t head_dim;     ///< Dimension of each attention head
    size_t hidden_size;  ///< Total hidden dimension size
    float dropout_prob;  ///< Dropout probability

    // Projection matrices
    Matrix query_proj;  ///< Query projection [hidden_size, num_heads * head_dim]
    Matrix key_proj;    ///< Key projection [hidden_size, num_kv_heads * head_dim]
    Matrix value_proj;  ///< Value projection [hidden_size, num_kv_heads * head_dim]
    Matrix output_proj; ///< Output projection [num_heads * head_dim, hidden_size]

    /**
     * @brief Repeats key/value heads to match query head count.
     * @param kv Key or value tensor to repeat
     * @param num_repeats Number of times to repeat each head
     * @return Matrix with repeated heads
     */
    Matrix repeat_kv_heads(const Matrix& kv, size_t num_repeats) const;

    /**
     * @brief Computes attention scores and weighted values using grouped heads.
     * @param Q Query matrix
     * @param K Key matrix
     * @param V Value matrix
     * @param mask Attention mask for causal or padding attention
     * @return Matrix of attention outputs
     */
    Matrix compute_grouped_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                                     const AttentionMask& mask) const;

  public:
    /**
     * @brief Constructs a grouped query attention layer.
     * @param hidden_size Size of input and output tensors
     * @param num_heads Number of query attention heads
     * @param num_kv_heads Number of key/value attention heads
     * @param head_dim Dimension of each attention head
     * @param dropout_prob Probability of dropout
     */
    GroupedQueryAttention(size_t hidden_size, size_t num_heads, size_t num_kv_heads,
                          size_t head_dim, float dropout_prob);

    /**
     * @brief Performs the forward pass of grouped query attention.
     * 
     * Projects inputs to query, key, and value spaces, computes attention
     * with grouped heads, and projects back to the original space.
     * 
     * @param x Input tensor of shape [batch_size, seq_len, hidden_size]
     * @param mask Attention mask for causal or padding attention
     * @param kv_cache Optional cache for key/value tensors in autoregressive generation
     * @return Output tensor of shape [batch_size, seq_len, hidden_size]
     */
    Matrix forward(const Matrix& x, const AttentionMask& mask,
                   const std::optional<KVCache>& kv_cache = std::nullopt);

    /**
     * @brief Gets the number of query attention heads.
     * @return Number of query heads
     */
    size_t get_num_heads() const {
        return num_heads;
    }

    /**
     * @brief Gets the number of key/value attention heads.
     * @return Number of key/value heads
     */
    size_t get_num_kv_heads() const {
        return num_kv_heads;
    }

    /**
     * @brief Gets the dimension of each attention head.
     * @return Head dimension
     */
    size_t get_head_dim() const {
        return head_dim;
    }
};