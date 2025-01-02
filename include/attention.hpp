#pragma once
#include "cache.hpp"
#include "components.hpp"
#include "tensor.hpp"
#include <optional>
using FloatVector = Vector;

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
  Matrix attention_scores;
  size_t hidden_size;
  float dropout_prob;
  bool use_gqa;
  size_t num_kv_heads;

  // Private helper methods
  Vector apply_rope(const Vector &x, size_t position) const;
  Matrix flash_attention(const Matrix &Q, const Matrix &K, const Matrix &V,
                         const AttentionMask &mask) const;
  Matrix standard_attention(const Matrix &Q, const Matrix &K, const Matrix &V,
                            const AttentionMask &mask);
  Tensor reshape_for_attention(const Matrix& x, size_t batch_size, 
                                size_t num_heads, size_t seq_len, size_t head_size) const;

  Matrix reshape_from_attention(const Tensor &x, size_t batch_size, size_t seq_len, size_t hidden_size) const;


Tensor compute_attention(const Matrix& Q, const Matrix& K, 
                      const Matrix& V, const AttentionMask& mask, 
                      size_t batch_size, size_t num_heads, 
                      size_t seq_len, size_t head_size);

  void validate_dimensions(const Matrix& grad_output, 
                         const Matrix& input,
                         const Matrix& target_dist) const {
       if (grad_output.cols() != hidden_size) {
           throw std::runtime_error("grad_output.cols (" + 
                                   std::to_string(grad_output.cols()) + 
                                   ") != hidden_size (" + 
                                   std::to_string(hidden_size) + ")");
       }
       if (input.cols() != hidden_size) {
           throw std::runtime_error("input.cols (" + 
                                   std::to_string(input.cols()) + 
                                   ") != hidden_size (" + 
                                   std::to_string(hidden_size) + ")");
       }
   }

  // Add private gradient computation methods
  Matrix compute_query_gradients(const Matrix& grad_output, const Matrix& input) const {
      // Q = input * Wq
      // dQ = grad_output * Wq^T
      return matmul(grad_output, query_proj.transpose());
  }
  
  Matrix compute_key_gradients(const Matrix& grad_output, const Matrix& input) const {
      // K = input * Wk
      // dK = grad_output * Wk^T
      return matmul(grad_output, key_proj.transpose());
  }
  
  Matrix compute_value_gradients(const Matrix& grad_output, const Matrix& input) const {
      // V = input * Wv
      // dV = grad_output * Wv^T
      return matmul(grad_output, value_proj.transpose());
  }
  
  Matrix combine_gradients(const Matrix& dQ, const Matrix& dK, const Matrix& dV) const {
      // Combine all gradients
      Matrix combined = dQ;
      combined += dK;
      combined += dV;
      return combined;
  }



  // Add compute_attention declaration
  Matrix compute_attention(const Matrix& Q, const Matrix& K, const Matrix& V, 
                         const AttentionMask& mask) {
       // Validate input dimensions
       if (Q.cols() != K.cols() || K.cols() != V.cols()) {
           throw std::runtime_error("Q, K, V dimension mismatch");
       }
       
       // Split heads
       size_t batch_size = Q.rows();
       size_t seq_len = Q.rows();
       size_t head_size = Q.cols() / num_heads;
       
       // Debug dimensions
       std::cout << "Attention dimensions:" << std::endl;
       std::cout << "batch_size: " << batch_size << ", seq_len: " << seq_len 
                 << ", head_size: " << head_size << ", num_heads: " << num_heads << std::endl;
       
       // Reshape Q, K, V to [batch_size, num_heads, seq_len, head_dim]
       std::cout << "Q.shape(): " << Q.shape() << std::endl;
       std::cout << "K.shape(): " << K.shape() << std::endl;
       std::cout << "V.shape(): " << V.shape() << std::endl;
       Tensor Q_reshaped = reshape_for_attention(Q, batch_size, num_heads, seq_len, head_size);
       Tensor K_reshaped = reshape_for_attention(K, batch_size, num_heads, seq_len, head_size);
       Tensor V_reshaped = reshape_for_attention(V, batch_size, num_heads, seq_len, head_size);
       std::cout << "exiting reshape_for_attention" << std::endl;
       // Compute attention scores with bounds checking
       Tensor scores = Tensor::safe_tensormul(Q_reshaped, K_reshaped.transpose({0, 1, 3, 2}));
       std::cout << "exiting safe_matmul" << std::endl;
       // Scale attention scores with temperature scaling
       const float temperature = std::sqrt(static_cast<float>(head_size));
       const float scale = 1.0f / temperature;
       const float MAX_SCORE = 10.0f;
       std::cout << "exiting temperature scaling" << std::endl;
       std::cout << "entering for loop" << std::endl;
       for(size_t i = 0; i < scores.size(); i++) {
           // Apply scaling with better numerical stability
           float val = scores.data()[i];
           if (std::abs(val) > 1e-6f) {  // Only scale non-zero values
               val *= scale;
               scores.data()[i] = std::clamp(val, -MAX_SCORE, MAX_SCORE);
           }
       }
       std::cout << "exiting for loop" << std::endl;
       // Apply mask if provided
       if (!mask.mask.empty()) {
           // Expand mask for all heads
           Matrix expanded_mask(scores.rows(), scores.cols());
           for (size_t h = 0; h < num_heads; h++) {
               for (size_t i = 0; i < mask.mask.rows(); i++) {
                   for (size_t j = 0; j < mask.mask.cols(); j++) {
                       expanded_mask(h * mask.mask.rows() + i, j) = mask.mask(i, j);
                   }
               }
           }
           apply_mask(scores, expanded_mask);
           std::cout << "exiting apply_mask" << std::endl;
       }
       
       std::cout << "Attention scores shape after mask: " 
                 << scores.rows() << "x" << scores.cols() << std::endl;
       std::cout << "Applying softmax" << std::endl;
       // Apply softmax with improved numerical stability
       apply_stable_softmax(scores);
       std::cout << "exiting apply_stable_softmax" << std::endl;
       std::cout << "entering safe_matmul" << std::endl;
       // Compute weighted sum with scaling
       Matrix attention = safe_matmul(scores, V_reshaped);
       std::cout << "exiting safe_matmul" << std::endl;
       // Add small epsilon to prevent exactly zero gradients
       const float EPSILON = 1e-6f;
       for(size_t i = 0; i < attention.size(); i++) {
           if (std::abs(attention.data()[i]) < EPSILON) {
               attention.data()[i] = attention.data()[i] < 0 ? -EPSILON : EPSILON;
           }
       }
       std::cout << "exiting for loop" << std::endl;
       std::cout << "entering reshape_from_attention" << std::endl;

       return reshape_from_attention(attention, batch_size, seq_len, hidden_size);
   }

   // Helper method for safe matrix multiplication
   Matrix safe_matmul(const Matrix& A, const Matrix& B) {
       if (A.cols() != B.rows()) {
           throw std::runtime_error("Matrix multiplication dimension mismatch: " +
                                  std::to_string(A.cols()) + " != " + 
                                  std::to_string(B.rows()));
       }
       return matmul(A, B);
   }

   
   Matrix reshape_from_attention(const Matrix& x, size_t batch_size, 
                               size_t seq_len, size_t hidden_size) const {
       // Reshape from [batch_size * num_heads, seq_len, head_size] to 
       // [batch_size, seq_len, hidden_size]
       std::cout << "entered reshape_from_attention" << std::endl;
       Matrix reshaped(batch_size, hidden_size);
       size_t head_size = hidden_size / num_heads;
       std::cout << "head_size: " << head_size << std::endl;    
       std::cout << "hidden_size: " << hidden_size << std::endl;
       std::cout << "num_heads: " << num_heads << std::endl;

       for (size_t b = 0; b < batch_size; b++) {
           for (size_t h = 0; h < num_heads; h++) {
               for (size_t s = 0; s < seq_len; s++) {
                   for (size_t d = 0; d < head_size; d++) {
                       size_t src_idx = (b * num_heads + h) * seq_len + s;
                       size_t tgt_idx = s * hidden_size + h * head_size + d;
                       reshaped.data()[tgt_idx] = x.data()[src_idx * head_size + d];
                   }
               }
           }
       }
       std::cout << "exiting reshape_from_attention" << std::endl;
       return reshaped;
   }
   
   void apply_mask(Matrix& scores, const Matrix& mask) const {
       std::cout << "Applying mask - scores shape: " << scores.rows() << "x" << scores.cols() 
                 << ", mask shape: " << mask.rows() << "x" << mask.cols() << std::endl;
       
       if (scores.rows() != mask.rows() || scores.cols() != mask.cols()) {
           throw std::runtime_error("Mask dimensions don't match attention scores: scores(" + 
                                   std::to_string(scores.rows()) + "," + 
                                   std::to_string(scores.cols()) + ") != mask(" + 
                                   std::to_string(mask.rows()) + "," + 
                                   std::to_string(mask.cols()) + ")");
       }
       
       for (size_t i = 0; i < scores.rows(); i++) {
           for (size_t j = 0; j < scores.cols(); j++) {
               if (mask(i,j) == 0.0f) {
                   scores(i,j) = -std::numeric_limits<float>::infinity();
               }
           }
       }
   }
   
   void apply_stable_softmax(Matrix& x) const {
       const float EPSILON = 1e-8f;  // Smaller epsilon for better precision
       const float MIN_SCORE = -1e4f;  // Lower minimum for better dynamic range
       
       for (size_t i = 0; i < x.rows(); i++) {
           // Find max excluding extreme negative values
           float max_val = MIN_SCORE;
           for (size_t j = 0; j < x.cols(); j++) {
               if (x(i,j) > MIN_SCORE) {
                   max_val = std::max(max_val, x(i,j));
               }
           }
           
           // Skip if all values are extremely negative
           if (max_val <= MIN_SCORE) {
               for (size_t j = 0; j < x.cols(); j++) {
                   x(i,j) = 1.0f / x.cols();  // Uniform distribution
               }
               continue;
           }
           
           // Compute exp and sum with numerical stability
           float sum = 0.0f;
           for (size_t j = 0; j < x.cols(); j++) {
               if (x(i,j) <= MIN_SCORE) {
                   x(i,j) = 0.0f;
               } else {
                   x(i,j) = std::exp(x(i,j) - max_val);
                   sum += x(i,j);
               }
           }
           
           // Normalize with epsilon to prevent zeros
           if (sum < EPSILON) {
               // If sum is too small, use uniform distribution
               for (size_t j = 0; j < x.cols(); j++) {
                   x(i,j) = 1.0f / x.cols();
               }
           } else {
               for (size_t j = 0; j < x.cols(); j++) {
                   x(i,j) = x(i,j) / sum;
                   // Add small noise to prevent exact zeros
                   if (x(i,j) > 0 && x(i,j) < EPSILON) {
                       x(i,j) += EPSILON * (1.0f + std::cos(static_cast<float>(j)));
                   }
               }
           }
       }
   }

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
  Matrix backward(const Matrix& grad_output,
                 const Matrix& input,
                 const Matrix& target_distribution);
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

  MultiHeadAttention(const MultiHeadAttention &other)
      : query_proj(other.query_proj), key_proj(other.key_proj),
        value_proj(other.value_proj), output_proj(other.output_proj),
        query_bias(other.query_bias), key_bias(other.key_bias),
        value_bias(other.value_bias), output_bias(other.output_bias),
        num_heads(other.num_heads), head_dim(other.head_dim),
        hidden_size(other.hidden_size),
        use_rope(other.use_rope), use_flash(other.use_flash),
        use_sliding_window(other.use_sliding_window),
        window_size(other.window_size), cos_cached(other.cos_cached),
        sin_cached(other.sin_cached) {}

  MultiHeadAttention &operator=(const MultiHeadAttention &other) {
    if (this != &other) {
      query_proj = other.query_proj;
      key_proj = other.key_proj;
      value_proj = other.value_proj;
      output_proj = other.output_proj;
      query_bias = other.query_bias;
      key_bias = other.key_bias;
      value_bias = other.value_bias;
      output_bias = other.output_bias;
      num_heads = other.num_heads;
      head_dim = other.head_dim;
      hidden_size = other.hidden_size;
      use_rope = other.use_rope;
      use_flash = other.use_flash;
      use_sliding_window = other.use_sliding_window;
      window_size = other.window_size;
      cos_cached = other.cos_cached;
      sin_cached = other.sin_cached;
    }
    return *this;
  }
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
      : MultiHeadAttention(), window_size(window_size_) {}
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