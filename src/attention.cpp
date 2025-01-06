#include "../include/attention.hpp"
#include "../include/cuda/cuda_utils.cuh"
#include "../include/gqa.hpp"
#include "../include/performance_metrics.hpp"
#include "../include/transformer.hpp"
#include "attention.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

extern PerformanceMetrics metrics;

Vector MultiHeadAttention::apply_rope(const Vector &x, size_t position) const {
  Vector result = x;
  // Apply rotary position embeddings
  for (size_t i = 0; i < x.size(); i += 2) {
    if (i + 1 >= x.size()) {
      std::cout << "Breaking at i=" << i << " (odd size)" << std::endl;
      break;
    }

    float x_i = x[i];
    float x_i1 = x[i + 1];

    // Each pair of elements belongs to a specific head and position within that
    // head
    size_t pair_idx = i / 2; // Index of the current pair
    size_t head_idx =
        pair_idx /
        (head_dim /
         2); // Which head (using half head_dim since we process pairs)
    size_t dim_idx =
        pair_idx % (head_dim / 2); // Position within head (using half head_dim)
    size_t cache_idx =
        head_idx * head_dim + dim_idx; // Correct: direct mapping to cache

    try {
      float cos_theta = get_cos_cached(position, cache_idx);
      float sin_theta = get_sin_cached(position, cache_idx);

      result[i] = x_i * cos_theta - x_i1 * sin_theta;
      result[i + 1] = x_i * sin_theta + x_i1 * cos_theta;
    } catch (const std::exception &e) {
      std::cout << "Error in RoPE application:" << std::endl;
      std::cout << "- Error message: " << e.what() << std::endl;
      std::cout << "- Current indices: pos=" << position
                << ", cache_idx=" << cache_idx << ", i=" << i << std::endl;
      throw;
    }
  }

  return result;
}

Matrix MultiHeadAttention::flash_attention(const Matrix &Q, const Matrix &K,
                                           const Matrix &V,
                                           const AttentionMask &mask) const {
  std::cout << "\n=== MultiHeadAttention::flash_attention START ==="
            << std::endl;
  const size_t seq_len = Q.rows();
  const size_t head_dim = Q.cols();

  // Block sizes based on hardware cache sizes
  const size_t Br = std::min(size_t(256), seq_len); // Q block size
  const size_t Bc = std::min(size_t(256), seq_len); // K/V block size

  Matrix O(seq_len, head_dim, 0.0f);
  std::vector<float> L(seq_len, 0.0f); // Scale factors
  std::vector<float> m(seq_len,
                       -std::numeric_limits<float>::infinity()); // Max values

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

Matrix MultiHeadAttention::forward(const Matrix &x, const AttentionMask &mask,
                                   const std::optional<KVCache> &kv_cache) {
  metrics.start_timer("attention_computation");

  size_t batch_size = x.rows() / mask.mask.rows();  // Derive batch size from input
  size_t seq_len = mask.mask.rows();
  
  // Project input to Q, K, V with batching
  Matrix Q = matmul(x, query_proj);
  Matrix K = matmul(x, key_proj);
  Matrix V = matmul(x, value_proj);

  // Handle batched attention computation
  if (use_flash_attention) {
    // Modify flash_attention to handle batches
    Matrix output(x.rows(), head_dim * num_heads, 0.0f);
    for (size_t b = 0; b < batch_size; b++) {
      size_t start_idx = b * seq_len;
      size_t end_idx = (b + 1) * seq_len;
      
      Matrix Q_batch = Q.block(start_idx, 0, seq_len, Q.cols());
      Matrix K_batch = K.block(start_idx, 0, seq_len, K.cols());
      Matrix V_batch = V.block(start_idx, 0, seq_len, V.cols());
      
      Matrix batch_output = flash_attention(Q_batch, K_batch, V_batch, mask);
      
      // Copy batch output back to full output
      for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < output.cols(); j++) {
          output(start_idx + i, j) = batch_output(i, j);
        }
      }
    }
    return output;
  } else {
    // Regular attention with batching
    return compute_attention(Q, K, V, mask, batch_size, num_heads, seq_len, head_dim);
  }
}

void MultiHeadAttention::save(std::ostream &os) const {
  std::cout << "\n=== MultiHeadAttention::save START ===" << std::endl;

  // Save dimensions and configuration
  std::cout << "Saving configuration..." << std::endl;
  std::cout << "- Number of heads: " << num_heads << std::endl;
  std::cout << "- Head dimension: " << head_dim << std::endl;
  os.write(reinterpret_cast<const char *>(&num_heads), sizeof(num_heads));
  os.write(reinterpret_cast<const char *>(&head_dim), sizeof(head_dim));

  // Save projection matrices
  std::cout << "\nSaving projection matrices..." << std::endl;
  std::cout << "Query projection shape: " << query_proj.rows() << "x"
            << query_proj.cols() << std::endl;
  query_proj.save(os);
  std::cout << "Key projection shape: " << key_proj.rows() << "x"
            << key_proj.cols() << std::endl;
  key_proj.save(os);
  std::cout << "Value projection shape: " << value_proj.rows() << "x"
            << value_proj.cols() << std::endl;
  value_proj.save(os);
  std::cout << "Output projection shape: " << output_proj.rows() << "x"
            << output_proj.cols() << std::endl;
  output_proj.save(os);

  std::cout << "=== MultiHeadAttention::save END ===\n" << std::endl;
}

std::unique_ptr<MultiHeadAttention>
MultiHeadAttention::load(std::istream &is, const TransformerConfig &config) {
  std::cout << "\n=== MultiHeadAttention::load START ===" << std::endl;

  // Read configuration
  std::cout << "Reading configuration..." << std::endl;
  size_t num_heads, head_dim;
  is.read(reinterpret_cast<char *>(&num_heads), sizeof(num_heads));
  is.read(reinterpret_cast<char *>(&head_dim), sizeof(head_dim));
  std::cout << "- Number of heads: " << num_heads << std::endl;
  std::cout << "- Head dimension: " << head_dim << std::endl;

  size_t hidden_size = num_heads * head_dim;

  // Use default values instead of config
  auto attention = std::make_unique<MultiHeadAttention>(
      hidden_size, num_heads, head_dim, config.dropout_rate,
      config.use_flash_attention, config.use_rope, config.use_sliding_window,
      config.window_size, config.use_gqa, num_heads, config.max_seq_length);

  // Load projection matrices
  std::cout << "\nLoading projection matrices..." << std::endl;
  attention->query_proj = Matrix::load(is);
  attention->key_proj = Matrix::load(is);
  attention->value_proj = Matrix::load(is);
  attention->output_proj = Matrix::load(is);

  // Validate loaded matrices
  std::cout << "\nValidating loaded matrices..." << std::endl;
  auto validate_matrix = [](const Matrix &m, const std::string &name) {
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

MultiHeadAttention::MultiHeadAttention(size_t hidden_size, size_t num_heads,
                                     size_t head_dim, float dropout_prob,
                                     bool use_flash_attn, bool use_rope,
                                     bool use_sliding_window, size_t window_size,
                                     bool use_gqa, size_t num_kv_heads,
                                     size_t max_seq_len)
    : num_heads(num_heads), head_dim(head_dim), hidden_size(hidden_size),
      dropout_prob(dropout_prob), use_flash_attention(use_flash_attn),
      use_rope(use_rope), use_sliding_window(use_sliding_window),
      window_size(window_size), use_gqa(use_gqa), num_kv_heads(num_kv_heads),
      max_seq_length(max_seq_len),
      // Initialize matrices with correct dimensions
      query_proj(hidden_size, num_heads * head_dim),
      key_proj(hidden_size, num_heads * head_dim),
      value_proj(hidden_size, num_heads * head_dim),
      output_proj(num_heads * head_dim, hidden_size),
      // Initialize bias vectors
      query_bias(hidden_size),
      key_bias(hidden_size),
      value_bias(hidden_size),
      output_bias(hidden_size),
      // Initialize gradients with same dimensions as their parameters
      query_proj_grad(hidden_size, num_heads * head_dim),
      key_proj_grad(hidden_size, num_heads * head_dim),
      value_proj_grad(hidden_size, num_heads * head_dim),
      output_proj_grad(num_heads * head_dim, hidden_size),
      query_bias_grad(hidden_size),
      key_bias_grad(hidden_size),
      value_bias_grad(hidden_size),
      output_bias_grad(hidden_size) {

  std::cout << "\n=== MultiHeadAttention::constructor START ===" << std::endl;

  // Print configuration
  std::cout << "Configuration:" << std::endl;
  std::cout << "- Hidden size: " << hidden_size << std::endl;
  std::cout << "- Number of heads: " << num_heads << std::endl;
  std::cout << "- Head dimension: " << head_dim << std::endl;
  std::cout << "- Dropout probability: " << dropout_prob << std::endl;
  std::cout << "- Use flash attention: " << std::boolalpha << use_flash_attention
            << std::endl;
  std::cout << "- Use RoPE: " << use_rope << std::endl;
  std::cout << "- Use sliding window: " << use_sliding_window << std::endl;
  std::cout << "- Window size: " << window_size << std::endl;
  std::cout << "- Use GQA: " << use_gqa << std::endl;
  std::cout << "- Number of KV heads: " << num_kv_heads << std::endl;

  // Validate input dimensions
  std::cout << "\nValidating dimensions..." << std::endl;
  if (hidden_size == 0 || num_heads == 0 || head_dim == 0) {
    throw std::runtime_error(
        "Invalid dimensions: hidden_size=" + std::to_string(hidden_size) +
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
  auto validate_matrix = [](const Matrix &m, const std::string &name) {
    if (m.empty()) {
      throw std::runtime_error(name + " is empty after initialization");
    }
    std::cout << name << " statistics:" << std::endl;
    std::cout << "- Shape: " << m.rows() << "x" << m.cols() << std::endl;
    std::cout << "- Range: [" << m.min() << ", " << m.max() << "]" << std::endl;
    if (std::isnan(m.min()) || std::isnan(m.max()) || std::isinf(m.min()) ||
        std::isinf(m.max())) {
      throw std::runtime_error("Invalid values in " + name +
                               " after initialization");
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

  initialize_rope_cache(max_seq_length, head_dim);
}

Matrix MultiHeadAttention::standard_attention(const Matrix &Q, const Matrix &K,
                                              const Matrix &V,
                                              const AttentionMask &mask) {
  std::cout << "\n=== MultiHeadAttention::standard_attention START ==="
            << std::endl;

  // Calculate dimensions
  size_t seq_len = Q.rows();
  size_t head_size = Q.cols() / num_heads;

  std::cout << "Dimensions:" << std::endl;
  std::cout << "- Sequence length: " << seq_len << std::endl;
  std::cout << "- Head size: " << head_size << std::endl;
  std::cout << "- Number of heads: " << num_heads << std::endl;

  // Always create a proper mask
  Matrix effective_mask(seq_len, seq_len,
                        1.0f); // Default to allowing all attention

  if (!mask.mask.empty()) {
    std::cout << "Original mask shape: " << mask.mask.rows() << "x"
              << mask.mask.cols() << std::endl;

    if (mask.mask.rows() != seq_len || mask.mask.cols() != seq_len) {
      std::cout << "WARNING: Mask size mismatch. Creating new mask..."
                << std::endl;
      // Create a new causal mask of the correct size
      AttentionMask new_mask = AttentionMask::create_causal_mask(seq_len);
      effective_mask = new_mask.mask;
    } else {
      effective_mask = mask.mask;
    }
  }

  std::cout << "Effective mask shape: " << effective_mask.rows() << "x"
            << effective_mask.cols() << std::endl;

  // First reshape inputs to separate heads
  Tensor Q_4d = reshape_for_attention(Q, 1, num_heads, seq_len, head_size);
  Tensor K_4d = reshape_for_attention(K, 1, num_heads, seq_len, head_size);
  Tensor V_4d = reshape_for_attention(V, 1, num_heads, seq_len, head_size);

  // Process each head separately
  Matrix final_output(seq_len, num_heads * head_size);

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

    // Compute attention scores for this head
    Matrix scores = matmul(Q_head, K_head.transpose()); // [seq_len, seq_len]

    // Scale scores
    float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    scores *= scale;

    // Apply mask if provided
    if (!effective_mask.empty()) {
      std::cout << "Applying mask for head " << h << std::endl;
      std::cout << "Scores shape: " << scores.rows() << "x" << scores.cols()
                << std::endl;
      std::cout << "Effective mask shape: " << effective_mask.rows() << "x"
                << effective_mask.cols() << std::endl;

      // Now the dimensions should match
      for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
          if (effective_mask(i, j) == 0.0f) {
            scores(i, j) = -1e6f;
          }
        }
      }
    }

    // Apply softmax
    for (size_t i = 0; i < scores.rows(); ++i) {
      float max_val = scores(i, 0);
      for (size_t j = 1; j < scores.cols(); ++j) {
        max_val = std::max(max_val, scores(i, j));
      }
      float sum = 0.0f;
      for (size_t j = 0; j < scores.cols(); ++j) {
        scores(i, j) = std::exp(scores(i, j) - max_val);
        sum += scores(i, j);
      }
      for (size_t j = 0; j < scores.cols(); ++j) {
        scores(i, j) /= (sum + 1e-6f);
      }
    }

    // Compute attention output for this head
    Matrix head_output = matmul(scores, V_head); // [seq_len, head_size]

    // Store this head's output in the final result
    for (size_t s = 0; s < seq_len; ++s) {
      for (size_t d = 0; d < head_size; ++d) {
        final_output(s, h * head_size + d) = head_output(s, d);
      }
    }
  }

  return final_output;
}

Matrix MultiHeadAttention::backward(const Matrix &grad_output,
                                    const Matrix &input,
                                    const Matrix &target_distribution) {
  validate_dimensions(grad_output, input, target_distribution);

  // Initialize gradients if not already done
  if (query_proj_grad.empty()) {
    query_proj_grad = Matrix(query_proj.rows(), query_proj.cols(), 0.0f);
    key_proj_grad = Matrix(key_proj.rows(), key_proj.cols(), 0.0f);
    value_proj_grad = Matrix(value_proj.rows(), value_proj.cols(), 0.0f);
    output_proj_grad = Matrix(output_proj.rows(), output_proj.cols(), 0.0f);
  }

  // Compute gradients for attention mechanism
  Matrix d_query = compute_query_gradients(grad_output, input);
  Matrix d_key = compute_key_gradients(grad_output, input);
  Matrix d_value = compute_value_gradients(grad_output, input);

  // Combine gradients
  Matrix d_input = combine_gradients(d_query, d_key, d_value);

  // Update projection gradients
  query_proj_grad += matmul(input.transpose(), d_query);
  key_proj_grad += matmul(input.transpose(), d_key);
  value_proj_grad += matmul(input.transpose(), d_value);
  output_proj_grad += matmul(grad_output.transpose(), input);

  // Update bias gradients
  Vector d_query_bias = d_query.row_sum();
  Vector d_key_bias = d_key.row_sum();
  Vector d_value_bias = d_value.row_sum();
  Vector d_output_bias = grad_output.row_sum();

  // Update bias gradients element by element
  for (size_t i = 0; i < query_bias_grad.size(); ++i) {
    query_bias_grad[i] += d_query_bias[i];
    key_bias_grad[i] += d_key_bias[i];
    value_bias_grad[i] += d_value_bias[i];
  }

  for (size_t i = 0; i < output_bias_grad.size(); ++i) {
    output_bias_grad[i] += d_output_bias[i];
  }

  return d_input;
}

Tensor MultiHeadAttention::reshape_for_attention(const Matrix &x,
                                                 size_t batch_size,
                                                 size_t num_heads,
                                                 size_t seq_len,
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

Matrix MultiHeadAttention::reshape_from_attention(const Tensor &x,
                                                  size_t batch_size,
                                                  size_t hidden_size) const {
  std::cout << "=== reshape_from_attention START ===" << std::endl;

  // Get dimensions from tensor
  const auto &dims = x.dims();
  size_t seq_len = dims[2]; // Third dimension is sequence length

  // Output should have shape (batch_size * seq_len, hidden_size)
  Matrix reshaped(batch_size * seq_len, hidden_size);

  // Reshape from [batch_size, num_heads, seq_len, head_dim] to [batch_size *
  // seq_len, hidden_size]
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

  std::cout << "Reshaped output dimensions: " << reshaped.rows() << "x"
            << reshaped.cols() << std::endl;
  std::cout << "=== reshape_from_attention END ===" << std::endl;

  return reshaped;
}

Matrix MultiHeadAttention::compute_attention(const Matrix &Q, const Matrix &K,
                                             const Matrix &V,
                                             const AttentionMask &mask) {
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
  Tensor Q_reshaped =
      reshape_for_attention(Q, 1, num_heads, seq_len, head_size);
  Tensor K_reshaped =
      reshape_for_attention(K, 1, num_heads, seq_len, head_size);
  Tensor V_reshaped =
      reshape_for_attention(V, 1, num_heads, seq_len, head_size);

  // Convert to matrices for computation while preserving effective dimensions
  Matrix Q_mat = Q_reshaped.to_matrix(); // [num_heads * seq_len, head_size]
  Matrix K_mat = K_reshaped.to_matrix(); // [num_heads * seq_len, head_size]
  Matrix V_mat = V_reshaped.to_matrix(); // [num_heads * seq_len, head_size]

  // Pre-compute scaling factor
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  
  // Use BLAS for matrix multiplications
  Matrix scores = matmul(Q_mat, K_mat.transpose());
  scores *= scale;
  
  // Vectorized mask application
  if (!mask.mask.empty()) {
      #pragma omp parallel for collapse(2)
      for (size_t i = 0; i < scores.rows(); i++) {
          for (size_t j = 0; j < scores.cols(); j++) {
              if (mask.mask(i, j) == 0.0f) {
                  scores.data()[i * scores.cols() + j] = -std::numeric_limits<float>::infinity();
              }
          }
      }
  }

  apply_stable_softmax(scores);

  // Compute attention output
  Matrix attention = matmul(scores, V_mat); // [num_heads * seq_len, head_size]

  // Reshape back to [seq_len, hidden_size]
  std::vector<unsigned long> dims = {static_cast<unsigned long>(1),
                                     static_cast<unsigned long>(num_heads),
                                     static_cast<unsigned long>(seq_len),
                                     static_cast<unsigned long>(head_size)};
  return reshape_from_attention(Tensor(attention, dims), seq_len, hidden_size);
}

AttentionMask AttentionMask::create_causal_mask(size_t size) {
  AttentionMask mask;
  mask.mask = Matrix(size, size, -std::numeric_limits<float>::infinity());

  // Create lower triangular matrix
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      mask.mask(i, j) = 0.0f;  // Allow attention to previous and current tokens
    }
  }
  return mask;
}

AttentionMask
AttentionMask::create_padding_mask(const std::vector<int> &lengths,
                                   size_t max_len) {
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

Tensor MultiHeadAttention::compute_attention(const Matrix &Q, const Matrix &K,
                                           const Matrix &V,
                                           const AttentionMask &mask,
                                           size_t batch_size,
                                           size_t num_heads, 
                                           size_t seq_len,
                                           size_t head_dim) {
    // Initialize output matrix for all batches
    Matrix output(batch_size * seq_len * num_heads, head_dim, 0.0f);
    
    // Process each batch separately
    for (size_t b = 0; b < batch_size; b++) {
        size_t batch_offset = b * seq_len * num_heads;
        
        // Extract batch slices
        Matrix Q_batch = Q.block(batch_offset, 0, seq_len * num_heads, head_dim);
        Matrix K_batch = K.block(batch_offset, 0, seq_len * num_heads, head_dim);
        Matrix V_batch = V.block(batch_offset, 0, seq_len * num_heads, head_dim);
        
        // Compute attention scores for this batch
        Matrix scores = matmul(Q_batch, K_batch.transpose());
        scores *= 1.0f / std::sqrt(static_cast<float>(head_dim));
        
        // Apply mask for this batch
        if (!mask.mask.empty()) {
            for (size_t h = 0; h < num_heads; h++) {
                for (size_t i = 0; i < seq_len; i++) {
                    for (size_t j = 0; j < seq_len; j++) {
                        size_t row = h * seq_len + i;
                        size_t col = h * seq_len + j;
                        if (mask.mask(i, j) == -std::numeric_limits<float>::infinity()) {
                            scores(row, col) = -std::numeric_limits<float>::infinity();
                        }
                    }
                }
            }
        }
        
        // Apply softmax per attention head
        for (size_t h = 0; h < num_heads; h++) {
            size_t start_idx = h * seq_len;
            size_t end_idx = (h + 1) * seq_len;
            apply_stable_softmax(scores, start_idx, end_idx);
        }
        
        // Compute attention output for this batch
        Matrix batch_output = matmul(scores, V_batch);
        
        // Copy to output
        for (size_t i = 0; i < batch_output.rows(); i++) {
            for (size_t j = 0; j < batch_output.cols(); j++) {
                output(batch_offset + i, j) = batch_output(i, j);
            }
        }
    }

    // Return as tensor with proper dimensions
    std::vector<unsigned long> dims = {
        static_cast<unsigned long>(batch_size),
        static_cast<unsigned long>(num_heads),
        static_cast<unsigned long>(seq_len),
        static_cast<unsigned long>(head_dim)
    };
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
  std::cout << "- cos_cached: " << cos_cached.rows() << "x" << cos_cached.cols()
            << std::endl;
  std::cout << "- sin_cached: " << sin_cached.rows() << "x" << sin_cached.cols()
            << std::endl;

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
    throw std::runtime_error(
        "RoPE cache access out of bounds: pos=" + std::to_string(pos) +
        ", dim=" + std::to_string(dim_idx));
  }
  return cos_cached(pos, dim_idx);
}

float MultiHeadAttention::get_sin_cached(size_t pos, size_t dim_idx) const {
  if (pos >= sin_cached.rows() || dim_idx >= sin_cached.cols()) {
    throw std::runtime_error(
        "RoPE cache access out of bounds: pos=" + std::to_string(pos) +
        ", dim=" + std::to_string(dim_idx));
  }
  return sin_cached(pos, dim_idx);
}

void MultiHeadAttention::apply_stable_softmax(Matrix& scores, size_t start_idx, size_t end_idx) {
    for (size_t i = start_idx; i < end_idx; i++) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < scores.cols(); j++) {
            max_val = std::max(max_val, scores(i, j));
        }
        
        float sum_exp = 0.0f;
        for (size_t j = 0; j < scores.cols(); j++) {
            scores(i, j) = std::exp(scores(i, j) - max_val);
            sum_exp += scores(i, j);
        }
        
        for (size_t j = 0; j < scores.cols(); j++) {
            scores(i, j) /= sum_exp;
        }
    }
}
