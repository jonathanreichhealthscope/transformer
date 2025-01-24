#include "../include/attention.hpp"
#include "../include/cuda/cuda_utils.cuh"
#include "../include/gqa.hpp"
#include "../include/performance_metrics.hpp"
#include "../include/transformer.hpp"
#include "../include/config.hpp"
#include "attention.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include "../include/half_precision.hpp"
#include "../include/cuda/attention_ops.cuh"
#include "../include/cuda/matrix_ops.cuh"

extern PerformanceMetrics metrics;

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
    std::cout << "\n=== MultiHeadAttention::flash_attention START ===" << std::endl;
    const size_t seq_len = Q.rows();
    const size_t head_dim = Q.cols();

    // Block sizes based on hardware cache sizes
    const size_t Br = std::min(size_t(256), seq_len); // Q block size
    const size_t Bc = std::min(size_t(256), seq_len); // K/V block size

    Matrix O(seq_len, head_dim, 0.0f);
    std::vector<float> L(seq_len, 0.0f);                                    // Scale factors
    std::vector<float> m(seq_len, -std::numeric_limits<float>::infinity()); // Max values

    // Iterate over blocks
    for (size_t kr = 0; kr < seq_len; kr += Br) {
        size_t kr_end = std::min(kr + Br, seq_len);

        for (size_t kc = 0; kc < seq_len; kc += Bc) {
            size_t kc_end = std::min(kc + Bc, seq_len);

            // Load Q, K, V blocks
            Matrix Qb = Q.block(kr, 0, kr_end - kr, head_dim);
            Matrix Kb = K.block(kc, 0, kc_end - kc, head_dim);
            Matrix Vb = V.block(kc, 0, kc_end - kc, head_dim);

            // Compute attention scores for this block
            Matrix S = matmul(Qb, Kb.transpose());
            S *= 1.0f / std::sqrt(static_cast<float>(head_dim));

            // Apply mask if needed
            if (!mask.mask.empty()) {
                for (size_t i = kr; i < kr_end; i++) {
                    for (size_t j = kc; j < kc_end; j++) {
                        if (mask.mask(i, j) == 0.0f) {
                            S(i - kr, j - kc) = -std::numeric_limits<float>::infinity();
                        }
                    }
                }
            }

            // Update running max and scale factors
            for (size_t i = 0; i < S.rows(); i++) {
                float mi = m[i + kr];
                float li = L[i + kr];

                for (size_t j = 0; j < S.cols(); j++) {
                    float sij = S(i, j);
                    if (sij > mi) {
                        float mi_new = sij;
                        float scale = std::exp(mi - mi_new);
                        li *= scale;
                        mi = mi_new;

                        // Scale existing output
                        for (size_t d = 0; d < head_dim; d++) {
                            O(i + kr, d) *= scale;
                        }
                    }

                    float pij = std::exp(sij - mi);
                    li += pij;

                    // Update output
                    for (size_t d = 0; d < head_dim; d++) {
                        O(i + kr, d) += pij * Vb(j, d);
                    }
                }

                m[i + kr] = mi;
                L[i + kr] = li;
            }
        }
    }

    // Normalize output
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t d = 0; d < head_dim; d++) {
            O(i, d) /= L[i];
        }
    }

    std::cout << "=== MultiHeadAttention::flash_attention END ===\n" << std::endl;
    return O;
}

Matrix MultiHeadAttention::forward(const Matrix& input, const AttentionMask& mask, 
                                 const std::optional<KVCache>& kv_cache) {
    try {
        // Project input to Q, K, V
        Matrix Q = query_proj.forward(input);
        Matrix K = key_proj.forward(input);
        Matrix V = value_proj.forward(input);
        
        // Get dimensions
        size_t batch_size = input.rows();
        size_t hidden_size = input.cols();  // This should be the model's hidden_size
        size_t seq_len = batch_size;  // For self-attention, sequence length equals batch size
        
#ifdef USE_CUDA
        try {
            // Use CUDA for attention computation
            Matrix scores(batch_size, batch_size);  // Full sequence length for attention
            
            // Batch compute attention scores and softmax
            cuda::compute_attention_scores(Q, K, scores, 1.0f / std::sqrt(head_dim), num_heads);
            if (mask) {
                scores += mask.value();
            }
            cuda::apply_softmax(scores);
            
            // Single synchronization point after main computation
            Matrix output(batch_size, hidden_size);
            cuda::attention_forward(Q, K, V, output, batch_size, num_heads, seq_len);
            CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();  // Single sync point
            
            // Final projection
            return output_proj.forward(output);
            
        } catch (const std::runtime_error& e) {
            std::cerr << "CUDA attention failed, falling back to CPU: " << e.what() << std::endl;
#endif
            // CPU fallback implementation
            Matrix scores = matmul(Q, K.transpose());
            scores *= (1.0f / std::sqrt(head_dim));

            if (mask) {
                scores += mask.value();
            }

            scores.apply_softmax();
            Matrix output = matmul(scores, V);
            
            // Ensure output has correct dimensions before projection
            if (output.cols() != hidden_size) {
                throw std::runtime_error("Attention output has wrong dimensions: " + 
                    std::to_string(output.cols()) + " vs expected " + std::to_string(hidden_size));
            }
            return output_proj.forward(output);
#ifdef USE_CUDA
        }
#endif
    } catch (const std::exception& e) {
        throw std::runtime_error("MultiHeadAttention forward failed: " + std::string(e.what()));
    }
}

void MultiHeadAttention::save(std::ostream& os) const {
    std::cout << "\n=== MultiHeadAttention::save START ===" << std::endl;

    // Save dimensions and configuration
    std::cout << "Saving configuration..." << std::endl;
    std::cout << "- Number of heads: " << num_heads << std::endl;
    std::cout << "- Head dimension: " << head_dim << std::endl;
    os.write(reinterpret_cast<const char*>(&num_heads), sizeof(num_heads));
    os.write(reinterpret_cast<const char*>(&head_dim), sizeof(head_dim));

    // Save projection matrices
    std::cout << "\nSaving projection matrices..." << std::endl;
    std::cout << "Query projection shape: " << query_proj.rows() << "x" << query_proj.cols()
              << std::endl;
    query_proj.save(os);
    std::cout << "Key projection shape: " << key_proj.rows() << "x" << key_proj.cols() << std::endl;
    key_proj.save(os);
    std::cout << "Value projection shape: " << value_proj.rows() << "x" << value_proj.cols()
              << std::endl;
    value_proj.save(os);
    std::cout << "Output projection shape: " << output_proj.rows() << "x" << output_proj.cols()
              << std::endl;
    output_proj.save(os);

    std::cout << "=== MultiHeadAttention::save END ===\n" << std::endl;
}

std::unique_ptr<MultiHeadAttention> MultiHeadAttention::load(std::istream& is, const TransformerConfig& config) {
    std::cout << "\n=== MultiHeadAttention::load START ===" << std::endl;

    // Read configuration
    std::cout << "Reading configuration..." << std::endl;
    size_t num_heads, head_dim;
    is.read(reinterpret_cast<char*>(&num_heads), sizeof(num_heads));
    is.read(reinterpret_cast<char*>(&head_dim), sizeof(head_dim));
    std::cout << "- Number of heads: " << num_heads << std::endl;
    std::cout << "- Head dimension: " << head_dim << std::endl;

    size_t hidden_size = num_heads * head_dim;

    // Create attention instance with config parameters
    auto attention = std::make_unique<MultiHeadAttention>(
        hidden_size, num_heads, head_dim, config.dropout_rate, 
        config.use_flash_attention, config.use_rope, 
        config.use_sliding_window, config.window_size, 
        config.use_gqa, num_heads, config.max_seq_length, config.use_fp16);

    // Load projection matrices
    std::cout << "\nLoading projection matrices..." << std::endl;
    attention->query_proj = Matrix::load(is);
    attention->key_proj = Matrix::load(is);
    attention->value_proj = Matrix::load(is);
    attention->output_proj = Matrix::load(is);

    // Validate loaded matrices
    std::cout << "\nValidating loaded matrices..." << std::endl;
    auto validate_matrix = [](const Matrix& m, const std::string& name) {
        if (m.empty()) {
            throw std::runtime_error(name + " is empty after loading");
        }
        std::cout << name << " statistics:" << std::endl;
        std::cout << "- Shape: " << m.rows() << "x" << m.cols() << std::endl;
        std::cout << "- Range: [" << m.min() << ", " << m.max() << "]" << std::endl;
        if (std::isnan(m.min()) || std::isnan(m.max()) || std::isinf(m.min()) ||
            std::isinf(m.max())) {
            throw std::runtime_error("Invalid values in " + name + " after loading");
        }
    };

    validate_matrix(attention->query_proj, "Query projection");
    validate_matrix(attention->key_proj, "Key projection");
    validate_matrix(attention->value_proj, "Value projection");
    validate_matrix(attention->output_proj, "Output projection");

    std::cout << "=== MultiHeadAttention::load END ===\n" << std::endl;
    return attention;
}

MultiHeadAttention::MultiHeadAttention(size_t hidden_size_, size_t num_heads_, size_t head_dim_,
                                       float dropout_prob_, bool use_flash_, bool use_rope_,
                                       bool use_sliding_window_, size_t window_size_, bool use_gqa_,
                                       size_t num_kv_heads_, size_t max_seq_length_, bool use_fp16)
    : num_heads(num_heads_), head_dim(head_dim_), hidden_size(hidden_size_),
      dropout_prob(dropout_prob_), use_flash(use_flash_), use_rope(use_rope_),
      use_sliding_window(use_sliding_window_), window_size(window_size_), use_gqa(use_gqa_),
      num_kv_heads(num_kv_heads_), max_seq_length(max_seq_length_),
      use_fp16_(use_fp16),
      // Initialize matrices with correct dimensions
      query_proj(Matrix(hidden_size_, hidden_size_)),
      key_proj(Matrix(hidden_size_, hidden_size_)),
      value_proj(Matrix(hidden_size_, hidden_size_)),
      output_proj(Matrix(hidden_size_, hidden_size_)),
      // Initialize bias vectors
      query_bias(FloatVector(hidden_size_ * num_heads_)),
      key_bias(FloatVector(hidden_size_ * num_heads_)),
      value_bias(FloatVector(hidden_size_ * num_heads_)), output_bias(FloatVector(hidden_size_)),
      // Initialize gradients with same dimensions as their parameters
      query_proj_grad(Matrix(hidden_size_, hidden_size_)),
      key_proj_grad(Matrix(hidden_size_, hidden_size_)),
      value_proj_grad(Matrix(hidden_size_, hidden_size_)),
      output_proj_grad(Matrix(hidden_size_, hidden_size_)),
      query_bias_grad(FloatVector(hidden_size_ * num_heads_)),
      key_bias_grad(FloatVector(hidden_size_ * num_heads_)),
      value_bias_grad(FloatVector(hidden_size_ * num_heads_)),
      output_bias_grad(FloatVector(hidden_size_)) {

    std::cout << "\n=== MultiHeadAttention::constructor START ===" << std::endl;

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
    float scale = std::sqrt(2.0f / (hidden_size + hidden_size));

    query_proj.randomize(-scale, scale);
    key_proj.randomize(-scale, scale);
    value_proj.randomize(-scale, scale);
    output_proj.randomize(-scale, scale);

    // Initialize biases with zero
    std::cout << "\nInitializing biases..." << std::endl;
    for (size_t i = 0; i < query_bias.size(); i++)
        query_bias[i] = 0.0f;
    for (size_t i = 0; i < key_bias.size(); i++)
        key_bias[i] = 0.0f;
    for (size_t i = 0; i < value_bias.size(); i++)
        value_bias[i] = 0.0f;
    for (size_t i = 0; i < output_bias.size(); i++)
        output_bias[i] = 0.0f;

    // Initialize gradients to zero
    for (size_t i = 0; i < query_proj_grad.rows(); i++) {
        for (size_t j = 0; j < query_proj_grad.cols(); j++) {
            query_proj_grad(i, j) = 0.0f;
            key_proj_grad(i, j) = 0.0f;
            value_proj_grad(i, j) = 0.0f;
            output_proj_grad(i, j) = 0.0f;
        }
    }

    for (size_t i = 0; i < query_bias_grad.size(); i++)
        query_bias_grad[i] = 0.0f;
    for (size_t i = 0; i < key_bias_grad.size(); i++)
        key_bias_grad[i] = 0.0f;
    for (size_t i = 0; i < value_bias_grad.size(); i++)
        value_bias_grad[i] = 0.0f;
    for (size_t i = 0; i < output_bias_grad.size(); i++)
        output_bias_grad[i] = 0.0f;

    // Validate initialization
    std::cout << "\nValidating initialization..." << std::endl;
    auto validate_matrix = [](const Matrix& m, const std::string& name) {
        if (m.empty()) {
            throw std::runtime_error(name + " is empty after initialization");
        }
        std::cout << name << " statistics:" << std::endl;
        std::cout << "- Shape: " << m.rows() << "x" << m.cols() << std::endl;
        std::cout << "- Range: [" << m.min() << ", " << m.max() << "]" << std::endl;
        if (std::isnan(m.min()) || std::isnan(m.max()) || std::isinf(m.min()) ||
            std::isinf(m.max())) {
            throw std::runtime_error("Invalid values in " + name + " after initialization");
        }
    };

    validate_matrix(query_proj, "Query projection");
    validate_matrix(key_proj, "Key projection");
    validate_matrix(value_proj, "Value projection");
    validate_matrix(output_proj, "Output projection");
    validate_matrix(query_proj_grad, "Query projection gradient");
    validate_matrix(key_proj_grad, "Key projection gradient");
    validate_matrix(value_proj_grad, "Value projection gradient");
    validate_matrix(output_proj_grad, "Output projection gradient");

    std::cout << "=== MultiHeadAttention::constructor END ===\n" << std::endl;

    initialize_rope_cache(max_seq_length_, head_dim);
}

Matrix MultiHeadAttention::standard_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                                              const AttentionMask& mask) {
    std::cout << "\n=== MultiHeadAttention::standard_attention START ===" << std::endl;

    // Calculate dimensions
    size_t seq_len = Q.rows();
    size_t head_size = Q.cols() / num_heads;

    std::cout << "Dimensions:" << std::endl;
    std::cout << "- Sequence length: " << seq_len << std::endl;
    std::cout << "- Head size: " << head_size << std::endl;
    std::cout << "- Number of heads: " << num_heads << std::endl;

    // Always create a proper mask
    Matrix effective_mask(seq_len, seq_len, 1.0f);

    if (!mask.mask.empty()) {
        effective_mask = mask.mask;
    }

    // First reshape inputs to separate heads
    Tensor Q_4d = reshape_for_attention(Q, 1, num_heads, seq_len, head_size);
    Tensor K_4d = reshape_for_attention(K, 1, num_heads, seq_len, head_size);
    Tensor V_4d = reshape_for_attention(V, 1, num_heads, seq_len, head_size);

    // Process each head separately with improved numerical stability
    Matrix final_output(seq_len, num_heads * head_size);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    const float max_score = 100.0f; // Prevent exp overflow

    for (size_t h = 0; h < num_heads; ++h) {
        // Extract matrices for current head
        Matrix Q_head(seq_len, head_size);
        Matrix K_head(seq_len, head_size);
        Matrix V_head(seq_len, head_size);

        // Copy data for current head
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t d = 0; d < head_size; ++d) {
                Q_head(s, d) = Q_4d.at(0, h, s, d);
                K_head(s, d) = K_4d.at(0, h, s, d);
                V_head(s, d) = V_4d.at(0, h, s, d);
            }
        }

        // Compute attention scores with numerical stability
        Matrix scores = matmul(Q_head, K_head.transpose());
        scores *= scale;

        // Apply mask and numerical stability improvements
        for (size_t i = 0; i < seq_len; ++i) {
            // Find max for numerical stability in softmax
            float row_max = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < seq_len; ++j) {
                if (effective_mask(i, j) == 0.0f) {
                    scores(i, j) = -std::numeric_limits<float>::infinity();
                } else {
                    scores(i, j) = std::min(scores(i, j), max_score);
                    row_max = std::max(row_max, scores(i, j));
                }
            }

            // Compute softmax with improved numerical stability
            float sum_exp = 0.0f;
            for (size_t j = 0; j < seq_len; ++j) {
                if (scores(i, j) != -std::numeric_limits<float>::infinity()) {
                    scores(i, j) = std::exp(scores(i, j) - row_max);
                    sum_exp += scores(i, j);
                } else {
                    scores(i, j) = 0.0f;
                }
            }

            // Normalize with careful handling of small values
            const float eps = 1e-6f;
            if (sum_exp < eps)
                sum_exp = eps;

            for (size_t j = 0; j < seq_len; ++j) {
                scores(i, j) /= sum_exp;
            }
        }

        // Compute attention output for this head
        Matrix head_output = matmul(scores, V_head);

        // Store this head's output in the final result
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t d = 0; d < head_size; ++d) {
                final_output(s, h * head_size + d) = head_output(s, d);
            }
        }
    }

    return final_output;
}

Matrix MultiHeadAttention::backward(const Matrix& grad_output, const Matrix& input,
                                    const Matrix& target_distribution) {
    // Get use_fp16 from member variable instead of global config
    static const bool use_fp16 = use_fp16_;
    
    Matrix grad = grad_output;
    Matrix input_matrix = input;
    
    if (use_fp16) {
        HalfPrecisionTraining::convert_to_fp16(grad);
        HalfPrecisionTraining::convert_to_fp16(input_matrix);
    }

    validate_dimensions(grad, input_matrix, target_distribution);

    // Initialize gradients if not already done
    if (query_proj_grad.empty()) {
        query_proj_grad = Matrix(query_proj.rows(), query_proj.cols(), 0.0f);
        key_proj_grad = Matrix(key_proj.rows(), key_proj.cols(), 0.0f);
        value_proj_grad = Matrix(value_proj.rows(), value_proj.cols(), 0.0f);
        output_proj_grad = Matrix(output_proj.rows(), output_proj.cols(), 0.0f);
    }

    // Compute gradients for attention mechanism
    Matrix d_query = compute_query_gradients(grad, input_matrix);
    std::cout << "d_query dimensions: " << d_query.rows() << "x" << d_query.cols() << std::endl;
    Matrix d_key = compute_key_gradients(grad, input_matrix);
    std::cout << "d_key dimensions: " << d_key.rows() << "x" << d_key.cols() << std::endl;
    Matrix d_value = compute_value_gradients(grad, input_matrix);
    std::cout << "d_value dimensions: " << d_value.rows() << "x" << d_value.cols() << std::endl;

    // Combine gradients
    Matrix d_input = combine_gradients(d_query, d_key, d_value);
    std::cout << "d_input dimensions: " << d_input.rows() << "x" << d_input.cols() << std::endl;
    
    // Update projection gradients
    std::cout << "Updating projection gradients..." << std::endl;
    
    // Project attention scores back to hidden dimensions for query
    Matrix d_query_hidden(d_query.rows(), input_matrix.cols());  // [seq_len x hidden_dim]
    cuda::matmul(d_query, query_proj, d_query_hidden);
    query_proj_grad += matmul(input_matrix.transpose(), d_query_hidden);
    std::cout << "query_proj_grad dimensions: " << query_proj_grad.rows() << "x" << query_proj_grad.cols() << std::endl;

    // Project attention scores back to hidden dimensions for key
    Matrix d_key_hidden(d_key.rows(), input_matrix.cols());
    cuda::matmul(d_key, key_proj, d_key_hidden);
    key_proj_grad += matmul(input_matrix.transpose(), d_key_hidden);
    std::cout << "key_proj_grad dimensions: " << key_proj_grad.rows() << "x" << key_proj_grad.cols() << std::endl;

    // Project attention scores back to hidden dimensions for value
    Matrix d_value_hidden(d_value.rows(), input_matrix.cols());
    cuda::matmul(d_value, value_proj, d_value_hidden);
    value_proj_grad += matmul(input_matrix.transpose(), d_value_hidden);
    std::cout << "value_proj_grad dimensions: " << value_proj_grad.rows() << "x" << value_proj_grad.cols() << std::endl;

    // For output projection, we already have grad in hidden dimensions
    output_proj_grad += matmul(grad.transpose(), input_matrix);  // This one is already correct
    std::cout << "output_proj_grad dimensions: " << output_proj_grad.rows() << "x" << output_proj_grad.cols() << std::endl;

    // Update bias gradients
    Vector d_query_bias = d_query.row_sum();
    std::cout << "d_query_bias dimensions: " << d_query_bias.size() << std::endl;   
    Vector d_key_bias = d_key.row_sum();
    std::cout << "d_key_bias dimensions: " << d_key_bias.size() << std::endl;
    Vector d_value_bias = d_value.row_sum();
    std::cout << "d_value_bias dimensions: " << d_value_bias.size() << std::endl;
    Vector d_output_bias = grad.row_sum();
    std::cout << "d_output_bias dimensions: " << d_output_bias.size() << std::endl;

    // Update bias gradients element by element
    std::cout << "Updating bias gradients..." << std::endl;
    for (size_t i = 0; i < query_bias_grad.size(); ++i) {
        query_bias_grad[i] += d_query_bias[i];
        key_bias_grad[i] += d_key_bias[i];
        value_bias_grad[i] += d_value_bias[i];
    }
    std::cout << "Updated query bias gradients" << std::endl;
    for (size_t i = 0; i < output_bias_grad.size(); ++i) {
        output_bias_grad[i] += d_output_bias[i];
    }

    // Convert gradients back to FP32 before returning
    if (use_fp16) {
        HalfPrecisionTraining::convert_to_fp32(grad);
    }

    return d_input;
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
                                             const AttentionMask& mask) {
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

    if (!mask.mask.empty()) {
        // Create expanded mask for all attention heads
        Matrix expanded_mask(scores.rows(), scores.cols(), 1.0f);

        // Repeat the mask for each attention head
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < seq_len; ++j) {
                    expanded_mask(h * seq_len + i, h * seq_len + j) = mask.mask(i, j);
                }
            }
        }

        std::cout << "Original mask shape: " << mask.mask.rows() << "x" << mask.mask.cols()
                  << std::endl;
        std::cout << "Expanded mask shape: " << expanded_mask.rows() << "x" << expanded_mask.cols()
                  << std::endl;
        std::cout << "Scores shape: " << scores.rows() << "x" << scores.cols() << std::endl;

        apply_mask(scores, expanded_mask);
    }

    apply_stable_softmax(scores);

    // Compute attention output
    Matrix attention = matmul(scores, V_mat); // [num_heads * seq_len, head_size]

    // Reshape back to [seq_len, hidden_size]
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

void MultiHeadAttention::initialize_rope_cache(size_t max_seq_len, size_t dim) {
    std::cout << "Initializing RoPE cache:" << std::endl;
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

Matrix MultiHeadAttention::compute_attention_scores(const Matrix& Q, const Matrix& K) {
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
            scores(i, j) = std::min(scores(i, j), max_score);
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

void MultiHeadAttention::apply_stable_softmax(Matrix& x) const {
    const float EPSILON = 1e-8f;
    const float TEMPERATURE = 0.7f;
    const float MIN_SCORE = -1e4f;

    for (size_t i = 0; i < x.rows(); i++) {
        // Find max for numerical stability
        float max_val = MIN_SCORE;
        for (size_t j = 0; j < x.cols(); j++) {
            if (x(i, j) > MIN_SCORE) {
                max_val = std::max(max_val, x(i, j));
            }
        }

        // Apply temperature scaling and compute sum
        float sum_exp = 0.0f;
        for (size_t j = 0; j < x.cols(); j++) {
            if (x(i, j) > MIN_SCORE) {
                x(i, j) = std::exp((x(i, j) - max_val) / TEMPERATURE);
                sum_exp += x(i, j);
            } else {
                x(i, j) = 0.0f;
            }
        }

        // Normalize
        if (sum_exp < EPSILON) {
            // If sum is too small, use uniform distribution
            float uniform_val = 1.0f / x.cols();
            for (size_t j = 0; j < x.cols(); j++) {
                x(i, j) = uniform_val;
            }
        } else {
            // Normal normalization
            for (size_t j = 0; j < x.cols(); j++) {
                x(i, j) /= sum_exp;
            }
        }
    }
}

Matrix MultiHeadAttention::compute_query_gradients(const Matrix& grad, const Matrix& input) {
    // Should produce (seq_len x seq_len) for attention scores
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
    int seq_len = input.rows();
    Matrix d_key(seq_len, seq_len);
    cuda::matmul(grad, key_proj.transpose(), d_key);
    
    std::cout << "compute_key_gradients dimensions:" << std::endl;
    std::cout << "grad: " << grad.rows() << "x" << grad.cols() << std::endl;
    std::cout << "key_proj: " << key_proj.rows() << "x" << key_proj.cols() << std::endl;
    std::cout << "d_key: " << d_key.rows() << "x" << d_key.cols() << std::endl;
    
    return d_key;
}

Matrix MultiHeadAttention::compute_value_gradients(const Matrix& grad, const Matrix& input) {
    int seq_len = input.rows();
    Matrix d_value(seq_len, seq_len);
    cuda::matmul(grad, value_proj.transpose(), d_value);
    
    std::cout << "compute_value_gradients dimensions:" << std::endl;
    std::cout << "grad: " << grad.rows() << "x" << grad.cols() << std::endl;
    std::cout << "value_proj: " << value_proj.rows() << "x" << value_proj.cols() << std::endl;
    std::cout << "d_value: " << d_value.rows() << "x" << d_value.cols() << std::endl;
    
    return d_value;
}

Matrix MultiHeadAttention::combine_gradients(const Matrix& d_query, const Matrix& d_key, const Matrix& d_value) {
    // Combine the gradients from Q, K, V projections
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
