#pragma once
#include "cache.hpp"
#include "components.hpp"
#include <optional>

class AttentionMask {
public:
  Matrix mask;
  static AttentionMask create_causal_mask(size_t size);
  static AttentionMask create_padding_mask(const std::vector<int> &lengths,
                                           size_t max_len);
  AttentionMask() = default;
};

class MultiHeadAttention {
private:
  Matrix query_proj;
  Matrix key_proj;
  Matrix value_proj;
  Matrix output_proj;
  FloatVector query_bias;
  FloatVector key_bias;
  FloatVector value_bias;
  FloatVector output_bias;
  size_t num_heads;
  size_t head_dim;
  bool use_rope;
  bool use_flash;
  bool use_sliding_window;
  size_t window_size;
  Matrix cos_cached;
  Matrix sin_cached;

  // Private helper methods
  Matrix apply_rope(const Matrix &x, size_t position) const;
  Matrix flash_attention(const Matrix &Q, const Matrix &K, const Matrix &V,
                         const AttentionMask &mask) const;
  Matrix standard_attention(const Matrix &Q, const Matrix &K, const Matrix &V,
                            const AttentionMask &mask) const;

public:
  virtual ~MultiHeadAttention() = default;
  MultiHeadAttention() = default;

  MultiHeadAttention(size_t hidden_size, size_t num_heads, size_t head_dim,
                     float dropout_prob = 0.1f, bool use_flash = true,
                     bool use_rope = true, bool use_sliding_window = false,
                     size_t window_size = 512, bool use_gqa = false,
                     size_t num_kv_heads = 0);

  Matrix forward(const Matrix &x, const AttentionMask &mask,
                 const std::optional<KVCache> &kv_cache = std::nullopt);
  Matrix backward(const Matrix &grad, const Matrix &input) const;
  Matrix backward_cuda(const Matrix &grad, const Matrix &input) const;
  void save(std::ostream &os) const;
  static std::unique_ptr<MultiHeadAttention> load(std::istream &is);
  friend class Transformer;

  std::vector<std::reference_wrapper<Matrix>> get_weights() {
    return {std::ref(query_proj), std::ref(key_proj), std::ref(value_proj),
            std::ref(output_proj)};
  }

  friend class TransformerLayer;

  FloatVector &getQueryBias() { return query_bias; }
  FloatVector &getKeyBias() { return key_bias; }
  FloatVector &getValueBias() { return value_bias; }
  FloatVector &getOutputBias() { return output_bias; }
};

// Add sliding window attention
class SlidingWindowAttention : public MultiHeadAttention {
private:
  size_t window_size;
  bool use_local_attention;

  void process_attention_window(const Matrix &Q, const Matrix &K,
                                const Matrix &V, Matrix &output, size_t start,
                                size_t end);

public:
  explicit SlidingWindowAttention(size_t window_size_ = 512)
      : window_size(window_size_) {}
  Matrix compute_local_attention(const Matrix &Q, const Matrix &K,
                                 const Matrix &V);
};

// Add sparse attention
class SparseAttention : public MultiHeadAttention {
private:
  std::vector<std::pair<int, int>> attention_patterns;
  float sparsity_threshold;

  Matrix compute_sparse_attention(const Matrix &Q, const Matrix &K,
                                  const Matrix &V) {
    // Implement sparse attention using custom patterns
    return Matrix();
  }
};