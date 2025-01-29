#pragma once
#include "vector.hpp"
#include "matrix.hpp"
#include "cache.hpp"
#include "components.hpp"
#include "tensor.hpp"
#include "config.hpp"
#include <optional>
#include <memory>
#include <vector>
using FloatVector = Vector;

// Utility functions for gradient computation
float compute_grad_norm(const Matrix& grad);
size_t count_params(const Matrix& param);

/**
 * @brief Class representing attention masks used in transformer attention mechanisms.
 * 
 * Attention masks are used to control which tokens can attend to which other tokens.
 * This class provides functionality for both causal (autoregressive) masking and
 * padding masking for variable length sequences.
 */
class AttentionMask {
  public:
    Matrix mask;  ///< The actual mask matrix where 0 indicates masked positions

    /**
     * @brief Creates a causal (triangular) mask for autoregressive attention.
     * @param size The sequence length to create the mask for
     * @return An AttentionMask object with a causal mask
     */
    static AttentionMask create_causal_mask(size_t size);

    /**
     * @brief Creates a padding mask for variable length sequences.
     * @param lengths Vector of actual sequence lengths
     * @param max_len Maximum sequence length to pad to
     * @return An AttentionMask object with a padding mask
     */
    static AttentionMask create_padding_mask(const std::vector<int>& lengths, size_t max_len);
    
    AttentionMask() = default;
    explicit operator bool() const { return has_mask_; }
    const Matrix& value() const { return mask_; }

    // Constructor taking a mask matrix
    explicit AttentionMask(const Matrix& mask) : mask_(mask), has_mask_(true) {}

    // Add is_masked method
    bool is_masked(size_t i, size_t j) const {
        return mask.empty() ? false : mask(i, j) == 0.0f;
    }

  private:
    Matrix mask_;
    bool has_mask_ = false;
};

class KVCache;

/**
 * @brief Implementation of Multi-Head Attention mechanism with various optimizations.
 * 
 * This class implements the core attention mechanism used in transformers, with support for:
 * - Multi-head attention with separate Q/K/V projections
 * - Grouped Query Attention (GQA)
 * - Rotary Position Embeddings (RoPE)
 * - Flash Attention optimization
 * - Sliding window attention
 * - Key-Value caching for efficient inference
 */
class MultiHeadAttention {
  public:
    virtual ~MultiHeadAttention() = default;
    MultiHeadAttention() = default;

    /**
     * @brief Constructs a multi-head attention layer with the specified parameters.
     * @param hidden_size_ Size of the input and output tensors
     * @param num_heads_ Number of attention heads
     * @param head_dim_ Dimension of each attention head
     * @param dropout_prob_ Dropout probability
     * @param use_flash_ Whether to use Flash Attention optimization
     * @param use_rope_ Whether to use rotary position embeddings
     * @param use_sliding_window_ Whether to use sliding window attention
     * @param window_size_ Size of the sliding window
     * @param use_gqa_ Whether to use grouped query attention
     * @param num_kv_heads_ Number of key/value heads for GQA
     * @param max_seq_length_ Maximum sequence length supported
     * @param use_fp16_ Whether to use fp16 for computation
     */
    MultiHeadAttention(size_t hidden_size_, size_t num_heads_, size_t head_dim_,
                       float dropout_prob_, bool use_flash_, bool use_rope_,
                       bool use_sliding_window_, size_t window_size_, bool use_gqa_,
                       size_t num_kv_heads_, size_t max_seq_length_, bool use_fp16);

    /**
     * @brief Performs the forward pass of the attention mechanism.
     * @param x Input tensor of shape [batch_size, seq_len, hidden_size]
     * @param mask Attention mask to prevent attending to certain positions
     * @param kv_cache Optional cache of key and value projections for efficient inference
     * @return Output tensor of shape [batch_size, seq_len, hidden_size]
     */
    Matrix forward(const Matrix& x, const AttentionMask& mask,
                   const std::optional<KVCache>& kv_cache = std::nullopt);

    /**
     * @brief Performs the backward pass to compute gradients.
     * @param grad_output Gradient of the loss with respect to the output
     * @param input Original input tensor
     * @param target_distribution Optional target attention distribution
     * @return Gradient with respect to the input
     */
    Matrix backward(const Matrix& grad_output, const Matrix& input,
                    const Matrix& target_distribution = Matrix());

    /**
     * @brief CUDA-accelerated version of the backward pass.
     * @param grad Gradient of the loss with respect to the output
     * @param input Original input tensor
     * @return Gradient with respect to the input
     */
    Matrix backward_cuda(const Matrix& grad, const Matrix& input) const;

    /**
     * @brief Saves the attention layer parameters to a stream.
     * @param os Output stream to save to
     */
    void save(std::ostream& os) const;

    /**
     * @brief Loads attention layer parameters from a stream.
     * @param is Input stream to load from
     * @param config Configuration object
     * @return Unique pointer to the loaded attention layer
     */
    static std::unique_ptr<MultiHeadAttention> load(std::istream& is, const TransformerConfig& config);

    /**
     * @brief Gets references to all trainable weight matrices.
     * @return Vector of references to weight matrices
     */
    std::vector<std::reference_wrapper<Matrix>> get_weights() {
        return {
            std::ref(params_.query_weights),
            std::ref(params_.key_weights),
            std::ref(params_.value_weights),
            std::ref(params_.output_weights)
        };
    }

    friend class Transformer;

    FloatVector& getQueryBias() { return params_.query_bias; }
    FloatVector& getKeyBias() { return params_.key_bias; }
    FloatVector& getValueBias() { return params_.value_bias; }
    FloatVector& getOutputBias() { return params_.output_bias; }

    MultiHeadAttention(const MultiHeadAttention& other)
        : params_(other.params_), grads_(other.grads_),
          num_heads(other.num_heads), head_dim(other.head_dim),
          hidden_size(other.hidden_size), dropout_prob(other.dropout_prob),
          use_rope(other.use_rope), use_flash(other.use_flash),
          use_sliding_window(other.use_sliding_window),
          window_size(other.window_size), use_gqa(other.use_gqa),
          num_kv_heads(other.num_kv_heads), max_seq_length(other.max_seq_length),
          use_fp16_(other.use_fp16_) {}

    MultiHeadAttention& operator=(const MultiHeadAttention& other) {
        if (this != &other) {
            params_ = other.params_;
            grads_ = other.grads_;
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
            max_seq_length = other.max_seq_length;
            use_fp16_ = other.use_fp16_;
        }
        return *this;
    }

    // Parameter structure to hold all weights and biases
    struct Parameters {
        Matrix query_weights;
        Matrix key_weights;
        Matrix value_weights;
        Matrix output_weights;
        FloatVector query_bias;
        FloatVector key_bias;
        FloatVector value_bias;
        FloatVector output_bias;
    };

    // Gradient structure to hold all gradients
    struct Gradients {
        Matrix query_grad;      // Gradient for query weights
        Matrix key_grad;        // Gradient for key weights
        Matrix value_grad;      // Gradient for value weights
        Matrix output_grad;     // Gradient for output weights
        FloatVector query_bias_grad;    // Gradient for query bias
        FloatVector key_bias_grad;      // Gradient for key bias
        FloatVector value_bias_grad;    // Gradient for value bias
        FloatVector output_bias_grad;   // Gradient for output bias
    };

    Parameters& parameters() { return params_; }
    Gradients& param_gradients() { return grads_; }
    const Parameters& parameters() const { return params_; }
    const Gradients& param_gradients() const { return grads_; }

    /**
     * @brief Computes attention scores between query and key matrices.
     * 
     * Implements the scaled dot-product attention mechanism:
     * scores = softmax(Q * K^T / sqrt(head_dim))
     * 
     * @param Q Query matrix [batch_size * num_heads, seq_len, head_dim]
     * @param K Key matrix [batch_size * num_kv_heads, seq_len, head_dim]
     * @return Attention scores [batch_size * num_heads, seq_len, seq_len]
     */
    Matrix compute_attention_scores(const Matrix& Q, const Matrix& K);

    /**
     * @brief Initialize the attention weights and biases
     * 
     * Initializes projection matrices with Xavier/Glorot initialization
     * and biases with small non-zero values for better training stability.
     */
    void initialize_weights();

    // Update accessor methods to use Parameters structure
    Matrix& get_query_weights() { return params_.query_weights; }
    Matrix& get_key_weights() { return params_.key_weights; }
    Matrix& get_value_weights() { return params_.value_weights; }
    Matrix& get_output_weights() { return params_.output_weights; }
    
    // Add const versions
    const Matrix& get_query_weights() const { return params_.query_weights; }
    const Matrix& get_key_weights() const { return params_.key_weights; }
    const Matrix& get_value_weights() const { return params_.value_weights; }
    const Matrix& get_output_weights() const { return params_.output_weights; }

    // Add the direct matrix version of forward
    Matrix forward(const Matrix& input, const Matrix& attention_mask);

    // Add the 5-parameter version of compute_attention_scores
    Matrix compute_attention_scores(const Matrix& Q, const Matrix& K, 
                                  const Matrix& attention_mask,
                                  size_t batch_size, size_t seq_len);

  private:
    Parameters params_;
    Gradients grads_;
    
    // Configuration parameters
    size_t num_heads;         ///< Number of attention heads
    size_t head_dim;         ///< Dimension of each attention head
    size_t hidden_size;       ///< Size of input and output tensors
    float dropout_prob;       ///< Dropout probability
    bool use_rope;           ///< Whether to use rotary position embeddings
    bool use_flash;          ///< Whether to use Flash Attention
    bool use_sliding_window; ///< Whether to use sliding window attention
    size_t window_size;      ///< Size of attention window
    bool use_gqa;           ///< Whether to use grouped query attention
    size_t num_kv_heads;     ///< Number of key/value heads for GQA
    size_t max_seq_length;   ///< Maximum sequence length supported
    bool use_fp16_;         ///< Whether to use FP16 computation

    // RoPE caches
    Matrix cos_cached;       ///< Cached cosine values for RoPE
    Matrix sin_cached;       ///< Cached sine values for RoPE

    // Private helper methods
    Vector apply_rope(const Vector& x, size_t position) const;
    Matrix flash_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                           const AttentionMask& mask) const;
    Matrix standard_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                              const AttentionMask& mask);
    Tensor reshape_for_attention(const Matrix& x, size_t batch_size, size_t num_heads,
                                 size_t seq_len, size_t head_size) const;

    // Change the inline definition to just a declaration
    Matrix reshape_from_attention(const Tensor& x, size_t seq_len, size_t hidden_size) const;

    Tensor compute_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                             const AttentionMask& mask, size_t batch_size, size_t num_heads,
                             size_t seq_len, size_t head_size);

    void validate_dimensions(const Matrix& grad_output, const Matrix& input,
                             const Matrix& target_dist) const {
        if (grad_output.cols() != hidden_size) {
            throw std::runtime_error("grad_output.cols (" + std::to_string(grad_output.cols()) +
                                     ") != hidden_size (" + std::to_string(hidden_size) + ")");
        }
        if (input.cols() != hidden_size) {
            throw std::runtime_error("input.cols (" + std::to_string(input.cols()) +
                                     ") != hidden_size (" + std::to_string(hidden_size) + ")");
        }
    }

    // Update method declarations to match implementations
    Matrix compute_query_gradients(const Matrix& grad, const Matrix& input);
    Matrix compute_key_gradients(const Matrix& grad, const Matrix& input);
    Matrix compute_value_gradients(const Matrix& grad, const Matrix& input);
    Matrix combine_gradients(const Matrix& d_query, const Matrix& d_key, const Matrix& d_value);
    Matrix compute_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                           const AttentionMask& mask) const;

    // Helper method for safe matrix multiplication
    Matrix safe_matmul(const Matrix& A, const Matrix& B) {
        if (A.cols() != B.rows()) {
            throw std::runtime_error("Matrix multiplication dimension mismatch: " +
                                     std::to_string(A.cols()) + " != " + std::to_string(B.rows()));
        }
        return matmul(A, B);
    }

    void apply_mask(Matrix& scores, const Matrix& mask) const {
        std::cout << "Applying mask - scores shape: " << scores.rows() << "x" << scores.cols()
                  << ", mask shape: " << mask.rows() << "x" << mask.cols() << std::endl;

        if (scores.rows() != mask.rows() || scores.cols() != mask.cols()) {
            throw std::runtime_error(
                "Mask dimensions don't match attention scores: scores(" +
                std::to_string(scores.rows()) + "," + std::to_string(scores.cols()) + ") != mask(" +
                std::to_string(mask.rows()) + "," + std::to_string(mask.cols()) + ")");
        }

        for (size_t i = 0; i < scores.rows(); i++) {
            for (size_t j = 0; j < scores.cols(); j++) {
                if (mask(i, j) == 0.0f) {
                    scores(i, j) = -std::numeric_limits<float>::infinity();
                }
            }
        }
    }

    // Move implementation to source file
    void apply_stable_softmax(Matrix& x) const;

    // Add these new methods to handle Tensors directly
    void apply_mask(Tensor& scores, const Matrix& mask) const {
        Matrix scores_mat = scores.to_matrix();
        apply_mask(scores_mat, mask);
        scores = Tensor(scores_mat,
                        {scores.dims()[0], scores.dims()[1], scores.dims()[2], scores.dims()[3]});
    }

    void apply_stable_softmax(Tensor& scores) const {
        Matrix scores_mat = scores.to_matrix();
        apply_stable_softmax(scores_mat);
        scores = Tensor(scores_mat,
                        {scores.dims()[0], scores.dims()[1], scores.dims()[2], scores.dims()[3]});
    }

    // RoPE helper functions
    /**
     * @brief Applies rotary position embeddings to Q/K matrices.
     * 
     * RoPE improves the model's ability to capture positional information
     * by encoding positions through rotation in vector space.
     * 
     * @param matrix Input Q or K matrix to apply RoPE to
     * @param offset Position offset for cached generation
     */
    void apply_rotary_embeddings(Matrix& matrix, size_t offset = 0);

    /**
     * @brief Initializes or updates RoPE caches.
     * @param max_seq_len Length of sequence to cache embeddings for
     * @param dim_idx Dimension index for cached generation
     */
    void initialize_rope_cache(size_t max_seq_len, size_t dim_idx);
    float get_cos_cached(size_t pos, size_t dim_idx) const;
    float get_sin_cached(size_t pos, size_t dim_idx) const;
};

// Add sliding window attention
class SlidingWindowAttention : public MultiHeadAttention {
  private:
    size_t window_size;
    bool use_local_attention;
    size_t head_dim;  // Add head dimension member

    /**
     * @brief Process attention for a single window
     * @param Q Query matrix for current position
     * @param K Key matrix for local window
     * @param V Value matrix for local window
     * @param output Output matrix to update
     * @param pos Current position in sequence
     * @param window_start Start position of current window
     */
    void process_attention_window(const Matrix& Q, const Matrix& K, const Matrix& V,
                                Matrix& output, size_t pos, size_t window_start);

  public:
    /**
     * @brief Construct a sliding window attention layer
     * @param window_size_ Size of the attention window
     * @param head_dim_ Dimension of each attention head
     * @param use_local_attention_ Whether to use local attention only
     */
    explicit SlidingWindowAttention(size_t window_size_ = 512, size_t head_dim_ = 64,
                                  bool use_local_attention_ = true)
        : MultiHeadAttention(), window_size(window_size_), head_dim(head_dim_),
          use_local_attention(use_local_attention_) {}

    /**
     * @brief Compute attention with local sliding windows
     * @param Q Query matrix
     * @param K Key matrix
     * @param V Value matrix
     * @return Output matrix after local attention
     */
    Matrix compute_local_attention(const Matrix& Q, const Matrix& K, const Matrix& V);
};

// Add sparse attention
class SparseAttention : public MultiHeadAttention {
  private:
    std::vector<std::pair<int, int>> attention_patterns;
    float sparsity_threshold;

    Matrix compute_sparse_attention(const Matrix& Q, const Matrix& K, const Matrix& V) {
        // Implement sparse attention using custom patterns
        return Matrix();
    }
};