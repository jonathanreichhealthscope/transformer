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
public:
  virtual ~MultiHeadAttention() = default;
  MultiHeadAttention() = default;

  MultiHeadAttention(size_t hidden_size_, 
                    size_t num_heads_, 
                    size_t head_dim_, 
                    float dropout_prob_,
                    bool use_flash_, 
                    bool use_rope_,
                    bool use_sliding_window_, 
                    size_t window_size_,
                    bool use_gqa_, 
                    size_t num_kv_heads_,
                    size_t max_seq_length_);

  Matrix forward(const Matrix &x, const AttentionMask &mask,
                 const std::optional<KVCache> &kv_cache = std::nullopt);
  Matrix backward(const Matrix& grad_output,
                 const Matrix& input,
                 const Matrix& target_distribution);
  Matrix backward_cuda(const Matrix &grad, const Matrix &input) const;
  void save(std::ostream &os) const;
  static std::unique_ptr<MultiHeadAttention> load(std::istream &is, const class TransformerConfig& config);
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
        query_proj_grad(other.query_proj_grad), key_proj_grad(other.key_proj_grad),
        value_proj_grad(other.value_proj_grad), output_proj_grad(other.output_proj_grad),
        query_bias_grad(other.query_bias_grad), key_bias_grad(other.key_bias_grad),
        value_bias_grad(other.value_bias_grad), output_bias_grad(other.output_bias_grad),
        num_heads(other.num_heads), head_dim(other.head_dim),
        hidden_size(other.hidden_size), dropout_prob(other.dropout_prob),
        use_rope(other.use_rope), use_flash(other.use_flash),
        use_sliding_window(other.use_sliding_window),
        window_size(other.window_size), use_gqa(other.use_gqa),
        num_kv_heads(other.num_kv_heads),
        cos_cached(other.cos_cached), sin_cached(other.sin_cached) {}

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
      query_proj_grad = other.query_proj_grad;
      key_proj_grad = other.key_proj_grad;
      value_proj_grad = other.value_proj_grad;
      output_proj_grad = other.output_proj_grad;
      query_bias_grad = other.query_bias_grad;
      key_bias_grad = other.key_bias_grad;
      value_bias_grad = other.value_bias_grad;
      output_bias_grad = other.output_bias_grad;
      num_heads = other.num_heads;
      head_dim = other.head_dim;
      hidden_size = other.hidden_size;
      dropout_prob = other.dropout_prob;
      use_rope = other.use_rope;
      use_flash = other.use_flash;
      use_sliding_window = other.use_sliding_window;
      window_size = other.window_size;
      use_gqa = other.use_gqa;
      num_kv_heads = other.num_kv_heads;
      cos_cached = other.cos_cached;
      sin_cached = other.sin_cached;
    }
    return *this;
  }

  struct Parameters {
      std::vector<std::reference_wrapper<Matrix>> matrices;
      std::vector<std::reference_wrapper<Vector>> vectors;

      // Add iterator support
      auto begin() { return matrices.begin(); }
      auto end() { return matrices.end(); }
      auto begin() const { return matrices.begin(); }
      auto end() const { return matrices.end(); }
  };

  Parameters& parameters() {
      params.matrices.clear();
      params.vectors.clear();
      
      // Matrix parameters
      params.matrices.emplace_back(query_proj);
      params.matrices.emplace_back(key_proj);
      params.matrices.emplace_back(value_proj);
      params.matrices.emplace_back(output_proj);
      
      // Vector parameters
      params.vectors.emplace_back(query_bias);
      params.vectors.emplace_back(key_bias);
      params.vectors.emplace_back(value_bias);
      params.vectors.emplace_back(output_bias);
      
      return params;
  }
  
  const Parameters& parameter_gradients() const {
      param_gradients.matrices.clear();
      param_gradients.vectors.clear();
      
      // Matrix gradients
      param_gradients.matrices.emplace_back(std::ref(const_cast<Matrix&>(query_proj_grad)));
      param_gradients.matrices.emplace_back(std::ref(const_cast<Matrix&>(key_proj_grad)));
      param_gradients.matrices.emplace_back(std::ref(const_cast<Matrix&>(value_proj_grad)));
      param_gradients.matrices.emplace_back(std::ref(const_cast<Matrix&>(output_proj_grad)));
      
      // Vector gradients
      param_gradients.vectors.emplace_back(std::ref(const_cast<Vector&>(query_bias_grad)));
      param_gradients.vectors.emplace_back(std::ref(const_cast<Vector&>(key_bias_grad)));
      param_gradients.vectors.emplace_back(std::ref(const_cast<Vector&>(value_bias_grad)));
      param_gradients.vectors.emplace_back(std::ref(const_cast<Vector&>(output_bias_grad)));
      
      return param_gradients;
  }

  Matrix compute_attention_scores(const Matrix& Q, const Matrix& K);

private:
  Parameters params;         // Trainable parameters
  mutable Parameters param_gradients;  // Parameter gradients

  // Gradients
  mutable Matrix query_proj_grad;
  mutable Matrix key_proj_grad;
  mutable Matrix value_proj_grad;
  mutable Matrix output_proj_grad;
  mutable FloatVector query_bias_grad;
  mutable FloatVector key_bias_grad;
  mutable FloatVector value_bias_grad;
  mutable FloatVector output_bias_grad;

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
  size_t max_seq_length;

  // Private helper methods
  Vector apply_rope(const Vector &x, size_t position) const;
  Matrix flash_attention(const Matrix &Q, const Matrix &K, const Matrix &V,
                         const AttentionMask &mask) const;
  Matrix standard_attention(const Matrix &Q, const Matrix &K, const Matrix &V,
                            const AttentionMask &mask);
  Tensor reshape_for_attention(const Matrix& x, size_t batch_size, 
                                size_t num_heads, size_t seq_len, size_t head_size) const;

  // Change the inline definition to just a declaration
  Matrix reshape_from_attention(const Tensor& x, size_t seq_len, size_t hidden_size) const;

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
                         const AttentionMask& mask);

   // Helper method for safe matrix multiplication
   Matrix safe_matmul(const Matrix& A, const Matrix& B) {
       if (A.cols() != B.rows()) {
           throw std::runtime_error("Matrix multiplication dimension mismatch: " +
                                  std::to_string(A.cols()) + " != " + 
                                  std::to_string(B.rows()));
       }
       return matmul(A, B);
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
   
   void apply_stable_softmax(Matrix& x) const;

   // Add these new methods to handle Tensors directly
   void apply_mask(Tensor& scores, const Matrix& mask) const {
       Matrix scores_mat = scores.to_matrix();
       apply_mask(scores_mat, mask);
       scores = Tensor(scores_mat, {scores.dims()[0], scores.dims()[1], scores.dims()[2], scores.dims()[3]});
   }

   void apply_stable_softmax(Tensor& scores) const {
       Matrix scores_mat = scores.to_matrix();
       apply_stable_softmax(scores_mat);
       scores = Tensor(scores_mat, {scores.dims()[0], scores.dims()[1], scores.dims()[2], scores.dims()[3]});
   }

   // RoPE helper functions
   void initialize_rope_cache(size_t max_seq_len, size_t dim);
   float get_cos_cached(size_t pos, size_t dim_idx) const;
   float get_sin_cached(size_t pos, size_t dim_idx) const;
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