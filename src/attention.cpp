#include "../include/attention.hpp"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/cuda/cuda_utils.cuh"
#include "../include/cuda/cuda_launch.cuh"
#include "../include/cuda/attention_ops.cuh"
#include "../include/cuda/matrix_ops.cuh"
#endif
#include "../include/gqa.hpp"
#include "../include/performance_metrics.hpp"
#include "../include/transformer.hpp"
#include "../include/config.hpp"
#include "../include/half_precision.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

extern PerformanceMetrics metrics;

// Initialize static members
Matrix MultiHeadAttention::cos_cached;
Matrix MultiHeadAttention::sin_cached;
bool MultiHeadAttention::rope_cache_initialized = false;

void MultiHeadAttention::initialize_static_rope_cache(size_t max_seq_len, size_t dim, size_t num_heads) {
    if (rope_cache_initialized) {
        return;  // Cache already initialized
    }

    std::cout << "Initializing static RoPE cache:" << std::endl;
    std::cout << "- max_seq_len: " << max_seq_len << std::endl;
    std::cout << "- dim: " << dim << std::endl;
    std::cout << "- num_heads: " << num_heads << std::endl;

    cos_cached = Matrix(max_seq_len, dim * num_heads);
    sin_cached = Matrix(max_seq_len, dim * num_heads);

    std::cout << "Created cache matrices:" << std::endl;
    std::cout << "- cos_cached: " << cos_cached.rows() << "x" << cos_cached.cols() << std::endl;
    std::cout << "- sin_cached: " << sin_cached.rows() << "x" << sin_cached.cols() << std::endl;

    for (size_t pos = 0; pos < max_seq_len; pos++) {
        for (size_t h = 0; h < num_heads; h++) {
            for (size_t i = 0; i < dim; i++) {
                size_t idx = h * dim + i;
                float theta = std::pow(10000.0f, -2.0f * i / dim);
                cos_cached(pos, idx) = std::cos(pos * theta);
                sin_cached(pos, idx) = std::sin(pos * theta);
            }
        }
    }

    rope_cache_initialized = true;
}

MultiHeadAttention::MultiHeadAttention(size_t hidden_size_, size_t num_heads_, size_t head_dim_,
                                     float dropout_prob_, bool use_flash_, bool use_rope_,
                                     bool use_sliding_window_, size_t window_size_, bool use_gqa_,
                                     size_t num_kv_heads_, size_t max_seq_length_, bool use_fp16)
    : num_heads(num_heads_), head_dim(head_dim_), hidden_size(hidden_size_),
      dropout_prob(dropout_prob_), use_flash(use_flash_), use_rope(use_rope_),
      use_sliding_window(use_sliding_window_), window_size(window_size_),
      use_gqa(use_gqa_), num_kv_heads(num_kv_heads_), max_seq_length(max_seq_length_),
      use_fp16_(use_fp16) {

    std::cout << "\n=== MultiHeadAttention::constructor START ===" << std::endl;

    // Initialize matrices with correct dimensions
    params_.query_weights = Matrix(hidden_size_, hidden_size_);
    params_.key_weights = Matrix(hidden_size_, hidden_size_);
    params_.value_weights = Matrix(hidden_size_, hidden_size_);
    params_.output_weights = Matrix(hidden_size_, hidden_size_);

    // Initialize bias vectors
    params_.query_bias = FloatVector(hidden_size_ * num_heads_);
    params_.key_bias = FloatVector(hidden_size_ * num_heads_);
    params_.value_bias = FloatVector(hidden_size_ * num_heads_);
    params_.output_bias = FloatVector(hidden_size_);

    // Initialize gradients
    grads_.query_grad = Matrix(hidden_size_, hidden_size_);
    grads_.key_grad = Matrix(hidden_size_, hidden_size_);
    grads_.value_grad = Matrix(hidden_size_, hidden_size_);
    grads_.output_grad = Matrix(hidden_size_, hidden_size_);
    grads_.query_bias_grad = FloatVector(hidden_size_ * num_heads_);
    grads_.key_bias_grad = FloatVector(hidden_size_ * num_heads_);
    grads_.value_bias_grad = FloatVector(hidden_size_ * num_heads_);
    grads_.output_bias_grad = FloatVector(hidden_size_);

    // Print configuration
    std::cout << "Configuration:" << std::endl;
    std::cout << "- Hidden size: " << hidden_size << std::endl;
    std::cout << "- Number of heads: " << num_heads << std::endl;
    std::cout << "- Head dimension: " << head_dim << std::endl;
    std::cout << "- Dropout probability: " << dropout_prob << std::endl;
    std::cout << "- Use flash attention: " << std::boolalpha << use_flash << std::endl;
    std::cout << "- Use RoPE: " << use_rope << std::endl;
    std::cout << "- Use sliding window: " << use_sliding_window << std::endl;
    std::cout << "- Window size: " << window_size << std::endl;
    std::cout << "- Use GQA: " << use_gqa << std::endl;
    std::cout << "- Number of KV heads: " << num_kv_heads << std::endl;

    // Validate input dimensions
    std::cout << "\nValidating dimensions..." << std::endl;
    if (hidden_size == 0 || num_heads == 0 || head_dim == 0) {
        throw std::runtime_error("Invalid dimensions: hidden_size=" + std::to_string(hidden_size) +
                               ", num_heads=" + std::to_string(num_heads) +
                               ", head_dim=" + std::to_string(head_dim));
    }

    if (hidden_size % num_heads != 0) {
        throw std::runtime_error("hidden_size must be divisible by num_heads");
    }
    std::cout << "Dimension validation passed" << std::endl;

    // Initialize weights with Xavier/Glorot initialization
    std::cout << "\nInitializing weights..." << std::endl;
    initialize_weights();

    // Initialize RoPE cache if needed and not already initialized
    if (use_rope) {
        initialize_static_rope_cache(max_seq_length_, head_dim, num_heads);
    }

    std::cout << "=== MultiHeadAttention::constructor END ===\n" << std::endl;
}

Vector MultiHeadAttention::apply_rope(const Vector& x, size_t position) const {
    Vector result = x;
    // Apply rotary position embeddings
    for (size_t i = 0; i < x.size(); i += 2) {
        if (i + 1 >= x.size()) {
            std::cout << "Breaking at i=" << i << " (odd size)" << std::endl;
            break;
        }

        float x_i = x[i];
        float x_i1 = x[i + 1];

        // Each pair of elements belongs to a specific head and position within that head
        size_t pair_idx = i / 2; // Index of the current pair
        size_t head_idx =
            pair_idx / (head_dim / 2); // Which head (using half head_dim since we process pairs)
        size_t dim_idx = pair_idx % (head_dim / 2); // Position within head (using half head_dim)
        size_t cache_idx = head_idx * head_dim + dim_idx; // Correct: direct mapping to cache

        try {
            float cos_theta = get_cos_cached(position, cache_idx);
            float sin_theta = get_sin_cached(position, cache_idx);

            result[i] = x_i * cos_theta - x_i1 * sin_theta;
            result[i + 1] = x_i * sin_theta + x_i1 * cos_theta;
        } catch (const std::exception& e) {
            std::cout << "Error in RoPE application:" << std::endl;
            std::cout << "- Error message: " << e.what() << std::endl;
            std::cout << "- Current indices: pos=" << position << ", cache_idx=" << cache_idx
                      << ", i=" << i << std::endl;
            throw;
        }
    }

    return result;
}

Matrix MultiHeadAttention::flash_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                                         const AttentionMask& mask) const {
    std::cout << "=== MultiHeadAttention::flash_attention START ===\n";
    
    const size_t seq_len = Q.rows();
    const size_t head_dim = Q.cols();
    Matrix O(seq_len, head_dim, 0.0f);
    std::vector<float> m(seq_len, -std::numeric_limits<float>::infinity());
    std::vector<float> L(seq_len, 0.0f);
    
    const float scale = 1.0f / std::sqrt(head_dim);  // Scaling factor for better numerical stability
    const float attn_clip = 5.0f;  // Clip attention scores
    
    // Process in blocks for better cache efficiency
    const size_t block_size = 64;  // Adjust based on cache size
    for (size_t kr = 0; kr < seq_len; kr += block_size) {
        const size_t k_end = std::min(kr + block_size, seq_len);
        
        // Compute attention scores for this block
        Matrix S(seq_len, k_end - kr);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = kr; j < k_end; j++) {
                float score = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    score += Q(i, d) * K(j, d);
                }
                score *= scale;
                
                // Apply mask if needed
                if (mask.is_masked(i, j)) {
                    score = -std::numeric_limits<float>::infinity();
                }
                
                // Clip attention scores for stability
                score = std::max(-attn_clip, std::min(attn_clip, score));
                S(i, j - kr) = score;
            }
        }
        
        // Update output with numerically stable softmax
        #pragma omp parallel for
        for (size_t i = 0; i < seq_len; i++) {
            float mi = m[i];
            float li = L[i];
            
            // Find max for numerical stability
            for (size_t j = 0; j < S.cols(); j++) {
                mi = std::max(mi, S(i, j));
            }
            
            // Compute softmax with improved stability
            std::vector<float> exp_scores(S.cols());
            float sum_exp = 0.0f;
            
            for (size_t j = 0; j < S.cols(); j++) {
                float e = std::exp(S(i, j) - mi);
                exp_scores[j] = e;
                sum_exp += e;
            }
            
            // Update output with normalized attention scores
            for (size_t j = 0; j < S.cols(); j++) {
                float attn_prob = exp_scores[j] / (sum_exp + 1e-6f);  // Add small epsilon
                for (size_t d = 0; d < head_dim; d++) {
                    O(i, d) += attn_prob * V(j + kr, d);
                }
            }
            
            m[i] = mi;
            L[i] = sum_exp;
        }
    }
    
    // Final normalization of output
    const float output_clip = 3.0f;  // Clip output values
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t d = 0; d < head_dim; d++) {
            O(i, d) = std::max(-output_clip, std::min(output_clip, O(i, d)));
        }
    }
    
    std::cout << "=== MultiHeadAttention::flash_attention END ===\n";
    return O;
}

Matrix MultiHeadAttention::forward(const Matrix& input, const AttentionMask& mask, const std::optional<KVCache>& kv_cache) {
    try {
        std::cout << "\n=== MultiHeadAttention::forward START ===" << std::endl;
        std::cout << "Input dims: " << input.rows() << "x" << input.cols() << std::endl;
        
        // Project input to Q, K, V
        Matrix Q = matmul(input, params_.query_weights);
        Matrix K = matmul(input, params_.key_weights);
        Matrix V = matmul(input, params_.value_weights);
        
        std::cout << "Q dims: " << Q.rows() << "x" << Q.cols() << std::endl;
        std::cout << "K dims: " << K.rows() << "x" << K.cols() << std::endl;
        std::cout << "V dims: " << V.rows() << "x" << V.cols() << std::endl;
        std::cout << "Query weights dims: " << params_.query_weights.rows() << "x" << params_.query_weights.cols() << std::endl;
        std::cout << "Key weights dims: " << params_.key_weights.rows() << "x" << params_.key_weights.cols() << std::endl;
        std::cout << "Value weights dims: " << params_.value_weights.rows() << "x" << params_.value_weights.cols() << std::endl;
        
        // Cache for backward pass
        GradientCheckpoint::cache_activation("query", Q);
        GradientCheckpoint::cache_activation("key", K);
        GradientCheckpoint::cache_activation("value", V);
        
        // Get dimensions
        size_t batch_size = input.rows();
        size_t hidden_size = input.cols();
        size_t seq_len = batch_size;
        
        std::cout << "batch_size: " << batch_size << std::endl;
        std::cout << "hidden_size: " << hidden_size << std::endl;
        std::cout << "seq_len: " << seq_len << std::endl;
        
        Matrix attention_output;
        if (use_flash) {
            attention_output = flash_attention(Q, K, V, mask);
        } else {
            Matrix attention_scores = compute_attention_scores(Q, K, mask);
            std::cout << "Attention scores dims: " << attention_scores.rows() << "x" << attention_scores.cols() << std::endl;
            GradientCheckpoint::cache_activation("attention_scores", attention_scores);
            attention_output = matmul(attention_scores, V);
        }
        std::cout << "Attention output dims: " << attention_output.rows() << "x" << attention_output.cols() << std::endl;
        
        // Final projection
        Matrix final_output = matmul(attention_output, params_.output_weights);
        std::cout << "Final output dims: " << final_output.rows() << "x" << final_output.cols() << std::endl;
        std::cout << "Output weights dims: " << params_.output_weights.rows() << "x" << params_.output_weights.cols() << std::endl;
        
        std::cout << "=== MultiHeadAttention::forward END ===\n" << std::endl;
        return final_output;
    } catch (const std::exception& e) {
        throw std::runtime_error("MultiHeadAttention forward failed: " + std::string(e.what()));
    }
}

Matrix MultiHeadAttention::compute_attention_scores(const Matrix& Q, const Matrix& K, const AttentionMask& mask) {
    Matrix scores = matmul(Q, K.transpose());

    // Scale scores with careful handling of numerical stability
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Clip extremely large values to prevent overflow
    const float max_score = 100.0f; // Prevent exp overflow

    for (size_t i = 0; i < scores.rows(); ++i) {
        // Find max for numerical stability in softmax
        float row_max = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < scores.cols(); ++j) {
            scores(i, j) *= scale;
            
            // Apply mask if needed
            if (mask.is_masked(i, j)) {
                scores(i, j) = -std::numeric_limits<float>::infinity();
            } else {
                scores(i, j) = std::min(scores(i, j), max_score);
            }
            row_max = std::max(row_max, scores(i, j));
        }

        // Compute softmax with improved numerical stability
        float sum_exp = 0.0f;
        for (size_t j = 0; j < scores.cols(); ++j) {
            scores(i, j) = std::exp(scores(i, j) - row_max);
            sum_exp += scores(i, j);
        }

        // Normalize with careful handling of small values
        const float eps = 1e-6f;
        if (sum_exp < eps)
            sum_exp = eps;

        for (size_t j = 0; j < scores.cols(); ++j) {
            scores(i, j) /= sum_exp;
        }
    }

    return scores;
}

Matrix MultiHeadAttention::backward(const Matrix& grad_output, const Matrix& input, const Matrix& target) {
    try {
        std::cout << "\n=== MultiHeadAttention::backward START ===" << std::endl;
        
        // Constants for gradient clipping and stability
        const float clip_threshold = 5.0f;  // Match global threshold
        const float eps = 1e-6f;
        
        Matrix output_weights_t = params_.output_weights.transpose();
        Matrix d_value = matmul(grad_output, output_weights_t);
        
        // Compute gradient norms
        float grad_norm = 0.0f;
        #pragma omp parallel for reduction(+:grad_norm)
        for (size_t i = 0; i < grad_output.size(); i++) {
            grad_norm += grad_output.data()[i] * grad_output.data()[i];
        }
        grad_norm = std::sqrt(grad_norm);
        
        // Compute scaling factor
        float scale = std::min(clip_threshold / (grad_norm + eps), 1.0f);
        scale = std::sqrt(scale);  // Softer scaling
        
        std::cout << "Attention gradient norm: " << grad_norm << std::endl;
        std::cout << "Attention scaling factor: " << scale << std::endl;
        
        // Scale gradients
        Matrix scaled_grad = grad_output;
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < scaled_grad.rows(); i++) {
            for (size_t j = 0; j < scaled_grad.cols(); j++) {
                scaled_grad(i, j) *= scale;
            }
        }
        
        // Accumulate weight gradients
        Matrix output_weight_grad = Matrix(params_.output_weights.rows(), params_.output_weights.cols(), 0.0f);
        Vector output_bias_grad = Vector(params_.output_weights.cols(), 0.0f);
        
        size_t batch_size = input.rows();
        for (size_t i = 0; i < batch_size; ++i) {
            Vector example_grad = scaled_grad.row(i);
            Vector example_value = input.row(i);
            
            Matrix example_grad_outer = outer_product(example_value, example_grad);
            output_weight_grad += example_grad_outer;
            output_bias_grad += example_grad;
        }
        
        // Update parameter gradients
        param_gradients().output_grad = output_weight_grad;
        param_gradients().output_bias_grad = output_bias_grad;
        
        // Compute input gradients for backward flow
        Matrix d_input = matmul(scaled_grad, params_.output_weights);
        
        std::cout << "=== MultiHeadAttention::backward END ===\n" << std::endl;
        return d_input;
        
    } catch (const std::exception& e) {
        std::cerr << "\nError in MultiHeadAttention::backward: " << e.what() << std::endl;
        throw;
    }
}

Matrix MultiHeadAttention::compute_query_gradients(const Matrix& grad, const Matrix& input) {
    // Original implementation
    int seq_len = input.rows();
    Matrix d_query(seq_len, seq_len);  // Attention scores dimensions
    
    // First compute attention score gradients
    cuda::compute_attention_scores(grad, input, d_query, 1.0f / std::sqrt(float(input.cols())), num_heads);
    
    std::cout << "compute_query_gradients dimensions:" << std::endl;
    std::cout << "grad: " << grad.rows() << "x" << grad.cols() << std::endl;
    std::cout << "input: " << input.rows() << "x" << input.cols() << std::endl;
    std::cout << "d_query: " << d_query.rows() << "x" << d_query.cols() << std::endl;

    return d_query;
}

Matrix MultiHeadAttention::compute_key_gradients(const Matrix& grad, const Matrix& input) {
    Matrix d_key(grad.rows(), grad.cols());
#ifdef USE_CUDA
    cuda::matmul(grad, params_.key_weights.transpose(), d_key);
#else
    d_key = matmul(grad, params_.key_weights.transpose());
#endif
    return d_key;
}

Matrix MultiHeadAttention::compute_value_gradients(const Matrix& grad, const Matrix& input) {
    Matrix d_value(grad.rows(), grad.cols());
#ifdef USE_CUDA
    cuda::matmul(grad, params_.value_weights.transpose(), d_value);
#else
    d_value = matmul(grad, params_.value_weights.transpose());
#endif
    return d_value;
}

Matrix MultiHeadAttention::combine_gradients(const Matrix& d_query, const Matrix& d_key, const Matrix& d_value) {
    // Original implementation
    Matrix combined = d_query;
    combined += d_key;
    combined += d_value;
    
    std::cout << "combine_gradients dimensions:" << std::endl;
    std::cout << "d_query: " << d_query.rows() << "x" << d_query.cols() << std::endl;
    std::cout << "d_key: " << d_key.rows() << "x" << d_key.cols() << std::endl;
    std::cout << "d_value: " << d_value.rows() << "x" << d_value.cols() << std::endl;
    std::cout << "combined: " << combined.rows() << "x" << combined.cols() << std::endl;
    
    return combined;
}

void MultiHeadAttention::initialize_weights() {
    // Xavier/Glorot initialization
    float scale = std::sqrt(2.0f / (hidden_size + head_dim));
    
    // Initialize projection matrices using parameter accessors
    params_.query_weights.initialize_random(scale);
    params_.key_weights.initialize_random(scale);
    params_.value_weights.initialize_random(scale);
    params_.output_weights.initialize_random(scale);
    
    // Initialize bias vectors with small values
    params_.query_bias.initialize_constant(0.01f);
    params_.key_bias.initialize_constant(0.01f);
    params_.value_bias.initialize_constant(0.01f);
    params_.output_bias.initialize_constant(0.01f);
}

float compute_grad_norm(const Matrix& grad) {
    float norm = 0.0f;
    #pragma omp parallel for reduction(+:norm)
    for (size_t i = 0; i < grad.rows(); ++i) {
        for (size_t j = 0; j < grad.cols(); ++j) {
            norm += grad(i, j) * grad(i, j);
        }
    }
    return std::sqrt(norm);
}

size_t count_params(const Matrix& param) {
    return param.rows() * param.cols();
}

float MultiHeadAttention::get_cos_cached(size_t pos, size_t dim_idx) const {
    if (pos >= cos_cached.rows() || dim_idx >= cos_cached.cols()) {
        throw std::runtime_error("RoPE cache access out of bounds: pos=" + std::to_string(pos) +
                                 ", dim=" + std::to_string(dim_idx));
    }
    return cos_cached(pos, dim_idx);
}

float MultiHeadAttention::get_sin_cached(size_t pos, size_t dim_idx) const {
    if (pos >= sin_cached.rows() || dim_idx >= sin_cached.cols()) {
        throw std::runtime_error("RoPE cache access out of bounds: pos=" + std::to_string(pos) +
                                 ", dim=" + std::to_string(dim_idx));
    }
    return sin_cached(pos, dim_idx);
}

void MultiHeadAttention::apply_stable_softmax(Matrix& x) const {
    // Process each row separately for proper attention distribution
    for (size_t row = 0; row < x.rows(); row++) {
        // Find max value in this row for numerical stability
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t col = 0; col < x.cols(); col++) {
            max_val = std::max(max_val, x(row, col));
        }

        // Subtract max value and compute exp for this row
        float row_sum = 0.0f;
        for (size_t col = 0; col < x.cols(); col++) {
            x(row, col) = std::exp(x(row, col) - max_val);
            row_sum += x(row, col);
        }

        // Check for numerical instability in this row
        if (row_sum < 1e-10) {
            std::cerr << "WARNING: Row " << row << " has near-zero softmax sum\n";
            // Fall back to uniform attention for this row only
            float uniform_val = 1.0f / x.cols();
            for (size_t col = 0; col < x.cols(); col++) {
                x(row, col) = uniform_val;
            }
            continue;
        }

        // Normalize this row
        for (size_t col = 0; col < x.cols(); col++) {
            x(row, col) /= row_sum;
        }
    }

    // Validate results
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < x.rows(); i++) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < x.cols(); j++) {
            min_val = std::min(min_val, x(i, j));
            max_val = std::max(max_val, x(i, j));
            row_sum += x(i, j);
        }
        if (std::abs(row_sum - 1.0f) > 1e-6) {
            std::cerr << "WARNING: Row " << i << " softmax sum = " << row_sum << "\n";
        }
    }
    
    std::cout << "Softmax output statistics:\n"
              << "Min: " << min_val << "\n"
              << "Max: " << max_val << "\n";
}

Tensor MultiHeadAttention::reshape_for_attention(const Matrix& x, size_t batch_size,
                                                 size_t num_heads, size_t seq_len,
                                                 size_t head_size) const {
    // Create a 4D tensor with shape [batch_size, num_heads, seq_len, head_size]
    Tensor reshaped(batch_size, num_heads, seq_len, head_size);

    // Copy and reshape the data
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                for (size_t d = 0; d < head_size; ++d) {
                    // Correct indexing for the input matrix
                    size_t flat_idx = s * x.cols() + h * head_size + d;
                    reshaped.at(b, h, s, d) = x.data()[flat_idx];
                }
            }
        }
    }
    return reshaped;
}

Matrix MultiHeadAttention::reshape_from_attention(const Tensor& x, size_t batch_size,
                                                  size_t hidden_size) const {
    std::cout << "=== reshape_from_attention START ===" << std::endl;

    // Get dimensions from tensor
    const auto& dims = x.dims();
    size_t seq_len = dims[2]; // Third dimension is sequence length

    // Output should have shape (batch_size * seq_len, hidden_size)
    Matrix reshaped(batch_size * seq_len, hidden_size);

    // Reshape from [batch_size, num_heads, seq_len, head_dim] to [batch_size * seq_len,
    // hidden_size]
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t d = 0; d < head_dim; ++d) {
                    // Calculate output position
                    size_t out_row = b * seq_len + s;
                    size_t out_col = h * head_dim + d;

                    // Get value from tensor
                    reshaped(out_row, out_col) = x.at(b, h, s, d);
                }
            }
        }
    }

    std::cout << "Reshaped output dimensions: " << reshaped.rows() << "x" << reshaped.cols()
              << std::endl;
    std::cout << "=== reshape_from_attention END ===" << std::endl;

    return reshaped;
}

Matrix MultiHeadAttention::compute_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                                          const AttentionMask& mask) const {
    // Validate input dimensions
    if (Q.cols() != K.cols() || K.cols() != V.cols()) {
        throw std::runtime_error("Q, K, V dimension mismatch");
    }

    size_t seq_len = Q.rows();
    size_t head_size = Q.cols() / num_heads;

    // Debug dimensions
    std::cout << "Attention dimensions:" << std::endl;
    std::cout << "Q: " << Q.rows() << "x" << Q.cols() << std::endl;
    std::cout << "K: " << K.rows() << "x" << K.cols() << std::endl;
    std::cout << "V: " << V.rows() << "x" << V.cols() << std::endl;
    std::cout << "seq_len: " << seq_len << ", head_size: " << head_size
              << ", num_heads: " << num_heads << std::endl;

    // Reshape maintaining [seq_len, hidden_size] as the basic shape
    Tensor Q_reshaped = reshape_for_attention(Q, 1, num_heads, seq_len, head_size);
    Tensor K_reshaped = reshape_for_attention(K, 1, num_heads, seq_len, head_size);
    Tensor V_reshaped = reshape_for_attention(V, 1, num_heads, seq_len, head_size);

    // Convert to matrices for computation while preserving effective dimensions
    Matrix Q_mat = Q_reshaped.to_matrix(); // [num_heads * seq_len, head_size]
    Matrix K_mat = K_reshaped.to_matrix(); // [num_heads * seq_len, head_size]
    Matrix V_mat = V_reshaped.to_matrix(); // [num_heads * seq_len, head_size]

    // Compute attention scores
    Matrix scores = matmul(Q_mat, K_mat.transpose()); // [num_heads * seq_len, seq_len]

    // Scale scores
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    scores *= scale;

    // Apply sliding window attention if enabled
    if (use_sliding_window) {
        const size_t half_window = window_size / 2;
        std::cout << "Applying sliding window attention with window size " << window_size << std::endl;
        
        #pragma omp parallel for collapse(2)
        for (size_t head = 0; head < num_heads; head++) {
            for (size_t i = 0; i < seq_len; i++) {
                size_t row_idx = head * seq_len + i;
                
                // Calculate window boundaries
                size_t window_start = (i >= half_window) ? i - half_window : 0;
                size_t window_end = std::min(i + half_window + 1, seq_len);
                
                // Mask out everything outside the window
                for (size_t j = 0; j < seq_len; j++) {
                    size_t col_idx = j;
                    if (j < window_start || j >= window_end) {
                        scores(row_idx, col_idx) = -std::numeric_limits<float>::infinity();
                    }
                }
            }
        }
    }

    // Apply attention mask if provided
    for (size_t i = 0; i < scores.rows(); i++) {
        for (size_t j = 0; j < scores.cols(); j++) {
            if (mask.is_masked(i % seq_len, j % seq_len)) {
                scores(i, j) = -std::numeric_limits<float>::infinity();
            }
        }
    }

    // Apply softmax with improved numerical stability
    apply_stable_softmax(scores);

    // Compute attention output
    Matrix attention = matmul(scores, V_mat);

    // Reshape back to original dimensions
    std::vector<unsigned long> dims = {
        static_cast<unsigned long>(1), static_cast<unsigned long>(num_heads),
        static_cast<unsigned long>(seq_len), static_cast<unsigned long>(head_size)};
    return reshape_from_attention(Tensor(attention, dims), seq_len, hidden_size);
}

AttentionMask AttentionMask::create_causal_mask(size_t size) {
    AttentionMask mask;
    mask.mask = Matrix(size, size, 0.0f);

    // Create lower triangular matrix
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size && j <= i; ++j) {
            mask.mask(static_cast<unsigned long>(i), static_cast<unsigned long>(j)) = 1.0f;
        }
    }
    return mask;
}

AttentionMask AttentionMask::create_padding_mask(const std::vector<int>& lengths, size_t max_len) {
    AttentionMask mask;
    size_t batch_size = lengths.size();
    mask.mask = Matrix(max_len, max_len, 0.0f);

    // Create padding mask
    for (size_t i = 0; i < max_len; ++i) {
        for (size_t j = 0; j < max_len; ++j) {
            // Allow attention up to the sequence length
            mask.mask(i, j) = (i < lengths[0] && j < lengths[0]) ? 1.0f : 0.0f;
        }
    }
    return mask;
}

Tensor MultiHeadAttention::compute_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                                             const AttentionMask& mask, size_t batch_size,
                                             size_t num_heads, size_t seq_len, size_t head_dim) {
    std::cout << "=== compute_attention START ===" << std::endl;

    // Validate input dimensions
    std::cout << "Validating dimensions..." << std::endl;
    std::cout << "Expected dimensions:" << std::endl;
    std::cout << "- batch_size: " << batch_size << std::endl;
    std::cout << "- num_heads: " << num_heads << std::endl;
    std::cout << "- seq_len: " << seq_len << std::endl;
    std::cout << "- head_dim: " << head_dim << std::endl;

    size_t expected_rows = batch_size * num_heads * seq_len;
    size_t expected_cols = head_dim;

    std::cout << "Q dimensions: " << Q.rows() << "x" << Q.cols() << std::endl;
    std::cout << "K dimensions: " << K.rows() << "x" << K.cols() << std::endl;
    std::cout << "V dimensions: " << V.rows() << "x" << V.cols() << std::endl;

    // Dimension validation...
    if (Q.rows() != expected_rows || Q.cols() != expected_cols) {
        throw std::runtime_error("Q dimensions mismatch");
    }
    if (K.rows() != expected_rows || K.cols() != expected_cols) {
        throw std::runtime_error("K dimensions mismatch");
    }
    if (V.rows() != expected_rows || V.cols() != expected_cols) {
        throw std::runtime_error("V dimensions mismatch");
    }

    // Initialize output matrix
    Matrix output(Q.rows(), V.cols(), 0.0f);

    // Block size for processing (adjust based on available memory)
    const size_t BLOCK_SIZE = 1024; // Process 1024 rows at a time
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Process attention in blocks
    for (size_t start_idx = 0; start_idx < Q.rows(); start_idx += BLOCK_SIZE) {
        size_t end_idx = std::min(start_idx + BLOCK_SIZE, Q.rows());
        size_t current_block_size = end_idx - start_idx;

        std::cout << "Processing block " << start_idx / BLOCK_SIZE + 1 << " of "
                  << (Q.rows() + BLOCK_SIZE - 1) / BLOCK_SIZE << std::endl;

        // Extract block of Q
        Matrix Q_block(current_block_size, Q.cols());
        for (size_t i = 0; i < current_block_size; ++i) {
            for (size_t j = 0; j < Q.cols(); ++j) {
                Q_block(i, j) = Q(start_idx + i, j);
            }
        }

        // Compute scores for this block
        Matrix scores = matmul(Q_block, K.transpose());
        scores *= scale_factor;

        // Apply mask for this block if provided
        if (!mask.mask.empty()) {
            for (size_t i = 0; i < current_block_size; ++i) {
                for (size_t j = 0; j < K.rows(); ++j) {
                    // Calculate original indices for masking
                    size_t orig_i = start_idx + i;
                    size_t batch_idx_i = orig_i / (num_heads * seq_len);
                    size_t head_idx_i = (orig_i % (num_heads * seq_len)) / seq_len;
                    size_t seq_idx_i = orig_i % seq_len;

                    size_t batch_idx_j = j / (num_heads * seq_len);
                    size_t head_idx_j = (j % (num_heads * seq_len)) / seq_len;
                    size_t seq_idx_j = j % seq_len;

                    // Apply mask only within same batch and head
                    if (batch_idx_i == batch_idx_j && head_idx_i == head_idx_j) {
                        if (mask.mask(seq_idx_i, seq_idx_j) == 0.0f) {
                            scores(i, j) = -std::numeric_limits<float>::infinity();
                        }
                    }
                }
            }
        }

        // Apply softmax row-wise
        for (size_t i = 0; i < current_block_size; ++i) {
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < scores.cols(); ++j) {
                max_val = std::max(max_val, scores(i, j));
            }

            float sum = 0.0f;
            for (size_t j = 0; j < scores.cols(); ++j) {
                scores(i, j) = std::exp(scores(i, j) - max_val);
                sum += scores(i, j);
            }

            for (size_t j = 0; j < scores.cols(); ++j) {
                scores(i, j) /= sum;
            }
        }

        // Compute output for this block
        Matrix block_output = matmul(scores, V);

        // Add block output to final output
        for (size_t i = 0; i < current_block_size; ++i) {
            for (size_t j = 0; j < V.cols(); ++j) {
                output(start_idx + i, j) = block_output(i, j);
            }
        }
    }

    // Create tensor with proper dimensions
    std::vector<unsigned long> dims = {
        static_cast<unsigned long>(batch_size), static_cast<unsigned long>(num_heads),
        static_cast<unsigned long>(seq_len), static_cast<unsigned long>(head_dim)};

    std::cout << "=== compute_attention END ===" << std::endl;
    return Tensor(output, dims);
}

void MultiHeadAttention::save(std::ostream& os) const {
    std::cout << "\n=== MultiHeadAttention::save START ===" << std::endl;

    // Save dimensions and configuration
    std::cout << "Saving configuration..." << std::endl;
    std::cout << "- Number of heads: " << num_heads << std::endl;
    std::cout << "- Head dimension: " << head_dim << std::endl;
    os.write(reinterpret_cast<const char*>(&num_heads), sizeof(num_heads));
    os.write(reinterpret_cast<const char*>(&head_dim), sizeof(head_dim));
    os.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
    os.write(reinterpret_cast<const char*>(&dropout_prob), sizeof(dropout_prob));
    os.write(reinterpret_cast<const char*>(&use_rope), sizeof(use_rope));
    os.write(reinterpret_cast<const char*>(&use_flash), sizeof(use_flash));
    os.write(reinterpret_cast<const char*>(&use_sliding_window), sizeof(use_sliding_window));
    os.write(reinterpret_cast<const char*>(&window_size), sizeof(window_size));
    os.write(reinterpret_cast<const char*>(&use_gqa), sizeof(use_gqa));
    os.write(reinterpret_cast<const char*>(&num_kv_heads), sizeof(num_kv_heads));
    os.write(reinterpret_cast<const char*>(&max_seq_length), sizeof(max_seq_length));
    os.write(reinterpret_cast<const char*>(&use_fp16_), sizeof(use_fp16_));

    // Save weight matrices
    std::cout << "Saving weight matrices..." << std::endl;
    params_.query_weights.save(os);
    params_.key_weights.save(os);
    params_.value_weights.save(os);
    params_.output_weights.save(os);

    // Save bias vectors
    std::cout << "Saving bias vectors..." << std::endl;
    os.write(reinterpret_cast<const char*>(params_.query_bias.data()), params_.query_bias.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(params_.key_bias.data()), params_.key_bias.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(params_.value_bias.data()), params_.value_bias.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(params_.output_bias.data()), params_.output_bias.size() * sizeof(float));

    std::cout << "=== MultiHeadAttention::save END ===\n" << std::endl;
}

std::unique_ptr<MultiHeadAttention> MultiHeadAttention::load(std::istream& is, const TransformerConfig& config) {
    std::cout << "\n=== MultiHeadAttention::load START ===" << std::endl;

    // Read configuration
    size_t num_heads, head_dim, hidden_size;
    float dropout_prob;
    bool use_rope, use_flash, use_sliding_window, use_gqa, use_fp16;
    size_t window_size, num_kv_heads, max_seq_length;

    is.read(reinterpret_cast<char*>(&num_heads), sizeof(num_heads));
    is.read(reinterpret_cast<char*>(&head_dim), sizeof(head_dim));
    is.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
    is.read(reinterpret_cast<char*>(&dropout_prob), sizeof(dropout_prob));
    is.read(reinterpret_cast<char*>(&use_rope), sizeof(use_rope));
    is.read(reinterpret_cast<char*>(&use_flash), sizeof(use_flash));
    is.read(reinterpret_cast<char*>(&use_sliding_window), sizeof(use_sliding_window));
    is.read(reinterpret_cast<char*>(&window_size), sizeof(window_size));
    is.read(reinterpret_cast<char*>(&use_gqa), sizeof(use_gqa));
    is.read(reinterpret_cast<char*>(&num_kv_heads), sizeof(num_kv_heads));
    is.read(reinterpret_cast<char*>(&max_seq_length), sizeof(max_seq_length));
    is.read(reinterpret_cast<char*>(&use_fp16), sizeof(use_fp16));

    std::cout << "Loaded configuration:" << std::endl;
    std::cout << "- Number of heads: " << num_heads << std::endl;
    std::cout << "- Head dimension: " << head_dim << std::endl;
    std::cout << "- Hidden size: " << hidden_size << std::endl;

    // Create attention instance
    auto attention = std::make_unique<MultiHeadAttention>(
        hidden_size, num_heads, head_dim,
        dropout_prob, use_flash, use_rope,
        use_sliding_window, window_size,
        use_gqa, num_kv_heads,
        max_seq_length, use_fp16
    );

    // Load weight matrices
    std::cout << "Loading weight matrices..." << std::endl;
    attention->params_.query_weights = Matrix::load(is);
    attention->params_.key_weights = Matrix::load(is);
    attention->params_.value_weights = Matrix::load(is);
    attention->params_.output_weights = Matrix::load(is);

    // Load bias vectors
    std::cout << "Loading bias vectors..." << std::endl;
    is.read(reinterpret_cast<char*>(attention->params_.query_bias.data()), attention->params_.query_bias.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(attention->params_.key_bias.data()), attention->params_.key_bias.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(attention->params_.value_bias.data()), attention->params_.value_bias.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(attention->params_.output_bias.data()), attention->params_.output_bias.size() * sizeof(float));

    std::cout << "=== MultiHeadAttention::load END ===\n" << std::endl;
    return attention;
}

