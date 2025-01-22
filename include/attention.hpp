#pragma once
#include "cache.hpp"
#include "components.hpp"
#include "tensor.hpp"
#include "config.hpp"
#include <optional>
using FloatVector = Vector;

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

  private:
    Matrix mask_;
    bool has_mask_ = false;
};

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
        return {std::ref(query_proj), std::ref(key_proj), std::ref(value_proj),
                std::ref(output_proj)};
    }

    friend class Transformer;

    FloatVector& getQueryBias() {
        return query_bias;
    }
    FloatVector& getKeyBias() {
        return key_bias;
    }
    FloatVector& getValueBias() {
        return value_bias;
    }
    FloatVector& getOutputBias() {
        return output_bias;
    }

    MultiHeadAttention(const MultiHeadAttention& other)
        : query_proj(other.query_proj), key_proj(other.key_proj), value_proj(other.value_proj),
          output_proj(other.output_proj), query_bias(other.query_bias), key_bias(other.key_bias),
          value_bias(other.value_bias), output_bias(other.output_bias),
          query_proj_grad(other.query_proj_grad), key_proj_grad(other.key_proj_grad),
          value_proj_grad(other.value_proj_grad), output_proj_grad(other.output_proj_grad),
          query_bias_grad(other.query_bias_grad), key_bias_grad(other.key_bias_grad),
          value_bias_grad(other.value_bias_grad), output_bias_grad(other.output_bias_grad),
          num_heads(other.num_heads), head_dim(other.head_dim), hidden_size(other.hidden_size),
          dropout_prob(other.dropout_prob), use_rope(other.use_rope), use_flash(other.use_flash),
          use_sliding_window(other.use_sliding_window), window_size(other.window_size),
          use_gqa(other.use_gqa), num_kv_heads(other.num_kv_heads),
          max_seq_length(other.max_seq_length), use_fp16_(other.use_fp16_) {}

    MultiHeadAttention& operator=(const MultiHeadAttention& other) {
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
            max_seq_length = other.max_seq_length;
            use_fp16_ = other.use_fp16_;
        }
        return *this;
    }

    struct Parameters {
        std::vector<std::reference_wrapper<Matrix>> matrices;
        std::vector<std::reference_wrapper<Vector>> vectors;

        // Add iterator support
        auto begin() {
            return matrices.begin();
        }
        auto end() {
            return matrices.end();
        }
        auto begin() const {
            return matrices.begin();
        }
        auto end() const {
            return matrices.end();
        }
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

  private:
    Parameters params;                  // Trainable parameters
    mutable Parameters param_gradients; // Parameter gradients

    // Add max_seq_length to member variables
    size_t max_seq_length;     ///< Maximum sequence length supported

    // Gradients
    mutable Matrix query_proj_grad;
    mutable Matrix key_proj_grad;
    mutable Matrix value_proj_grad;
    mutable Matrix output_proj_grad;
    mutable FloatVector query_bias_grad;
    mutable FloatVector key_bias_grad;
    mutable FloatVector value_bias_grad;
    mutable FloatVector output_bias_grad;

    // Projection matrices and their gradients
    Matrix query_proj;        ///< Query projection matrix [hidden_size, num_heads * head_dim]
    Matrix key_proj;          ///< Key projection matrix [hidden_size, num_kv_heads * head_dim]
    Matrix value_proj;        ///< Value projection matrix [hidden_size, num_kv_heads * head_dim]
    Matrix output_proj;       ///< Output projection matrix [num_heads * head_dim, hidden_size]
    
    // Bias vectors and their gradients
    FloatVector query_bias;   ///< Query projection bias
    FloatVector key_bias;     ///< Key projection bias
    FloatVector value_bias;   ///< Value projection bias
    FloatVector output_bias;  ///< Output projection bias
    
    // Configuration parameters
    size_t num_heads;         ///< Number of attention heads
    size_t head_dim;          ///< Dimension of each attention head
    size_t hidden_size;       ///< Size of input and output tensors
    float dropout_prob;       ///< Dropout probability
    bool use_rope;           ///< Whether to use rotary position embeddings
    bool use_flash;          ///< Whether to use Flash Attention
    bool use_sliding_window; ///< Whether to use sliding window attention
    size_t window_size;      ///< Size of attention window
    bool use_gqa;           ///< Whether to use grouped query attention
    size_t num_kv_heads;     ///< Number of key/value heads for GQA

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

    // Add use_fp16 as a member variable
    bool use_fp16_;
};

// Add sliding window attention
class SlidingWindowAttention : public MultiHeadAttention {
  private:
    size_t window_size;
    bool use_local_attention;

    void process_attention_window(const Matrix& Q, const Matrix& K, const Matrix& V, Matrix& output,
                                  size_t start, size_t end);

  public:
    explicit SlidingWindowAttention(size_t window_size_ = 512)
        : MultiHeadAttention(), window_size(window_size_) {}
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