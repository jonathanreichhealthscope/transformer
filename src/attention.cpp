#include "../include/attention.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

Vector MultiHeadAttention::apply_rope(const Vector &x, size_t position) const {
  std::cout << "entering MultiHeadAttention::apply_rope" << std::endl;
  Vector result = x;

  // Apply rotary position embeddings
  for (size_t i = 0; i < x.size(); i += 2) {
    if (i + 1 >= x.size()) break;

    float x_i = x[i];
    float x_i1 = x[i + 1];

    float cos_theta = cos_cached(position, i / 2);
    float sin_theta = sin_cached(position, i / 2);

    result[i] = x_i * cos_theta - x_i1 * sin_theta;
    result[i + 1] = x_i * sin_theta + x_i1 * cos_theta;
  }

  std::cout << "exiting MultiHeadAttention::apply_rope" << std::endl;
  return result;
}

Matrix MultiHeadAttention::flash_attention(const Matrix &Q, const Matrix &K,
                                           const Matrix &V,
                                           const AttentionMask &mask) const {
  std::cout << "entering MultiHeadAttention::flash_attention" << std::endl;
  const size_t seq_length = Q.rows();
  const size_t block_size = window_size;
  Matrix output(Q.rows(), V.cols(), 0.0f);

  // Process in blocks for better memory efficiency
  for (size_t b_start = 0; b_start < seq_length; b_start += block_size) {
    size_t b_end = std::min(b_start + block_size, seq_length);

    // Create block views
    Matrix K_block(b_end - b_start, K.cols());
    Matrix V_block(b_end - b_start, V.cols());

    // Copy block data
    for (size_t i = b_start; i < b_end; ++i) {
      for (size_t j = 0; j < K.cols(); ++j) {
        K_block(i - b_start, j) = K(i, j);
      }
      for (size_t j = 0; j < V.cols(); ++j) {
        V_block(i - b_start, j) = V(i, j);
      }
    }

    // Compute attention scores for this block
    Matrix scores = matmul(Q, K_block.transpose());
    scores *= 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Apply mask if provided
    if (!mask.mask.empty()) {
      for (size_t i = 0; i < scores.rows(); ++i) {
        for (size_t j = 0; j < scores.cols(); ++j) {
          if (mask.mask(i, j) == 0.0f) {
            scores(i, j) = -std::numeric_limits<float>::infinity();
          }
        }
      }
    }

    // Apply softmax
    scores.apply_softmax();

    // Compute weighted sum
    Matrix block_output = matmul(scores, V_block);

    // Add to output
    for (size_t i = 0; i < output.rows(); ++i) {
      for (size_t j = 0; j < output.cols(); ++j) {
        output(i, j) += block_output(i, j);
      }
    }
  }

  std::cout << "exiting MultiHeadAttention::flash_attention" << std::endl;
  return output;
}

Matrix MultiHeadAttention::forward(const Matrix &x, const AttentionMask &mask,
                                   const std::optional<KVCache> &kv_cache) {
  std::cout << "entering MultiHeadAttention::forward" << std::endl;
  // Debug input stats
  std::cout << "Input matrix x stats before projection:" << std::endl;
  print_matrix_stats(x);
  
  // Validate input dimensions
  if (x.cols() != query_proj.rows()) {
      throw std::runtime_error("Input dimension mismatch: x.cols (" + 
                              std::to_string(x.cols()) + 
                              ") != query_proj.rows (" + 
                              std::to_string(query_proj.rows()) + ")");
  }

  // Add numerical stability checks
  const float EPSILON = 1e-6f;
  const float MAX_VAL = 1e2f;
  
  // Project input to Q, K, V with dimension checks
  try {
      Matrix Q = matmul(x, query_proj);
      Matrix K = matmul(x, key_proj);
      Matrix V = matmul(x, value_proj);
      
      // Validate projected dimensions
      if (Q.cols() != head_dim * num_heads || 
          K.cols() != head_dim * num_heads || 
          V.cols() != head_dim * num_heads) {
          throw std::runtime_error("Projection dimension mismatch");
      }
      
      // Add bias with numerical stability
      auto safe_add_bias = [EPSILON, MAX_VAL](Matrix& m, const FloatVector& bias) {
          if (m.cols() != bias.size()) {
              throw std::runtime_error("Bias dimension mismatch");
          }
          for(size_t i = 0; i < m.rows(); i++) {
              for(size_t j = 0; j < m.cols(); j++) {
                  float val = m(i,j) + bias[j];
                  val = std::clamp(val, -MAX_VAL, MAX_VAL);
                  if (std::abs(val) < EPSILON) {
                      val = val < 0 ? -EPSILON : EPSILON;
                  }
                  m(i,j) = val;
              }
          }
      };
      
      safe_add_bias(Q, query_bias);
      safe_add_bias(K, key_bias);
      safe_add_bias(V, value_bias);

      std::cout << "After projection:" << std::endl;
      std::cout << "Q stats:" << std::endl;
      print_matrix_stats(Q);
      std::cout << "K stats:" << std::endl;
      print_matrix_stats(K);
      std::cout << "V stats:" << std::endl;
      print_matrix_stats(V);

      // Reshape for attention computation
      size_t batch_size = x.rows();
      size_t seq_len = x.rows();  // For self-attention, seq_len = batch_size
      
      // Validate attention mask dimensions if provided
      if (!mask.mask.empty() && 
          (mask.mask.rows() != seq_len || mask.mask.cols() != seq_len)) {
          throw std::runtime_error("Attention mask dimension mismatch");
      }

      std::cout << "exiting MultiHeadAttention::forward" << std::endl;
      return compute_attention(Q, K, V, mask);
  } catch (const std::exception& e) {
      std::cerr << "Error in attention forward pass: " << e.what() << std::endl;
      throw;
  }
}

void MultiHeadAttention::save(std::ostream &os) const {
  std::cout << "entering MultiHeadAttention::save" << std::endl;
  // Save dimensions and configuration
  os.write(reinterpret_cast<const char *>(&num_heads), sizeof(num_heads));
  os.write(reinterpret_cast<const char *>(&head_dim), sizeof(head_dim));

  // Save projection matrices
  query_proj.save(os);
  key_proj.save(os);
  value_proj.save(os);
  output_proj.save(os);
  std::cout << "exiting MultiHeadAttention::save" << std::endl;
}

std::unique_ptr<MultiHeadAttention> MultiHeadAttention::load(std::istream &is) {
  std::cout << "entering MultiHeadAttention::load" << std::endl;
  size_t num_heads, head_dim;
  is.read(reinterpret_cast<char *>(&num_heads), sizeof(num_heads));
  is.read(reinterpret_cast<char *>(&head_dim), sizeof(head_dim));

  auto attention =
      std::make_unique<MultiHeadAttention>(num_heads * head_dim, // hidden_size
                                           num_heads, head_dim);

  // Load projection matrices
  attention->query_proj = Matrix::load(is);
  attention->key_proj = Matrix::load(is);
  attention->value_proj = Matrix::load(is);
  attention->output_proj = Matrix::load(is);

  std::cout << "exiting MultiHeadAttention::load" << std::endl;
  return attention;
}

MultiHeadAttention::MultiHeadAttention(size_t hidden_size_, size_t num_heads_, 
                                     size_t head_dim_, float dropout_prob_,
                                     bool use_flash_, bool use_rope_,
                                     bool use_sliding_window_, size_t window_size_,
                                     bool use_gqa_, size_t num_kv_heads_)
    : query_proj(Matrix(hidden_size_, hidden_size_)),
      key_proj(Matrix(hidden_size_, hidden_size_)),
      value_proj(Matrix(hidden_size_, hidden_size_)),
      output_proj(Matrix(hidden_size_, hidden_size_)),
      query_bias(FloatVector(hidden_size_)),
      key_bias(FloatVector(hidden_size_)),
      value_bias(FloatVector(hidden_size_)),
      output_bias(FloatVector(hidden_size_)),
      num_heads(num_heads_),
      head_dim(head_dim_),
      hidden_size(hidden_size_),
      dropout_prob(dropout_prob_),
      use_flash(use_flash_),
      use_rope(use_rope_),
      use_sliding_window(use_sliding_window_),
      window_size(window_size_),
      use_gqa(use_gqa_),
      num_kv_heads(num_kv_heads_) {
  std::cout << "entering MultiHeadAttention constructor" << std::endl;
  // Initialize projection matrices
  query_proj = Matrix(hidden_size, num_heads * head_dim);
  key_proj = Matrix(hidden_size, num_heads * head_dim);
  value_proj = Matrix(hidden_size, num_heads * head_dim);
  output_proj = Matrix(num_heads * head_dim, hidden_size);

  // Use Xavier/Glorot initialization with proper scaling
  float q_scale = std::sqrt(2.0f / (hidden_size + head_dim * num_heads));
  float kv_scale = std::sqrt(2.0f / (hidden_size + head_dim));
  float out_scale = std::sqrt(2.0f / (hidden_size * 2));

  query_proj.randomize(-q_scale, q_scale);
  key_proj.randomize(-kv_scale, kv_scale);
  value_proj.randomize(-kv_scale, kv_scale);
  output_proj.randomize(-out_scale, out_scale);

  // Initialize biases to small positive values to prevent dead neurons
  const float BIAS_INIT = 0.01f;
  for(size_t i = 0; i < query_bias.size(); i++) query_bias[i] = BIAS_INIT;
  for(size_t i = 0; i < key_bias.size(); i++) key_bias[i] = BIAS_INIT;
  for(size_t i = 0; i < value_bias.size(); i++) value_bias[i] = BIAS_INIT;
  for(size_t i = 0; i < output_bias.size(); i++) output_bias[i] = BIAS_INIT;

  // Validate initialization
  if(query_proj.max() == 0.0f || key_proj.max() == 0.0f || value_proj.max() == 0.0f) {
    throw std::runtime_error("Attention projection matrices failed to initialize");
  }

  // Initialize RoPE buffers if needed
  if (use_rope) {
    // Fix: Use ceiling division to ensure we have enough columns
    size_t required_cols = (head_dim + 1) / 2; // Ceiling division
    cos_cached = Matrix(window_size, required_cols);
    sin_cached = Matrix(window_size, required_cols);

    std::cout << "Initializing RoPE buffers with dimensions: " << window_size
              << "x" << required_cols << std::endl;

    // Initialize RoPE angle cache
    for (size_t pos = 0; pos < window_size; ++pos) {
      for (size_t i = 0; i < required_cols; ++i) {
        float theta = pos / std::pow(10000.0f, (2.0f * i) / head_dim);
        cos_cached(pos, i) = std::cos(theta);
        sin_cached(pos, i) = std::sin(theta);
      }
    }
  }
  std::cout << "exiting MultiHeadAttention constructor" << std::endl;
}

Matrix MultiHeadAttention::standard_attention(const Matrix &Q, const Matrix &K,
                                              const Matrix &V,
                                              const AttentionMask &mask) {
  Matrix scores = matmul(Q, K.transpose());
  
  // Clamp extreme values in scores
  for(size_t i = 0; i < scores.size(); i++) {
    scores.data()[i] = std::clamp(scores.data()[i], -10.0f, 10.0f);
  }

  // Add numerical stability to attention scaling
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  scale = std::min(scale, 10.0f);  // Prevent too large scaling
  scores *= scale;

  if (!mask.mask.empty()) {
    for (size_t i = 0; i < scores.rows(); ++i) {
      for (size_t j = 0; j < scores.cols(); ++j) {
        if (mask.mask(i, j) == 0.0f) {
          scores(i, j) = -1e6f;  // Use finite value instead of infinity
        }
      }
    }
  }

  // Add numerical stability to softmax
  for (size_t i = 0; i < scores.rows(); ++i) {
    float max_val = scores(i, 0);
    for (size_t j = 1; j < scores.cols(); ++j) {
      max_val = std::max(max_val, scores(i, j));
    }
    float sum = 0.0f;
    const float epsilon = 1e-10f;
    for (size_t j = 0; j < scores.cols(); ++j) {
      scores(i, j) = std::exp(scores(i, j) - max_val);
      sum += scores(i, j);
    }
    sum = std::max(sum, epsilon);
    for (size_t j = 0; j < scores.cols(); ++j) {
      scores(i, j) /= sum;
    }
  }

  // Validate no NaN in output
  for(size_t i = 0; i < scores.size(); i++) {
    if(std::isnan(scores.data()[i])) {
      std::cerr << "NaN detected in attention scores!" << std::endl;
      scores.data()[i] = 0.0f;  // Replace NaN with zero
    }
  }

  // Store attention scores for backward pass
  attention_scores = scores;
  
  return matmul(scores, V);
}

Matrix MultiHeadAttention::backward(const Matrix& grad_output,
                                  const Matrix& input,
                                  const Matrix& target_distribution) {
    try {
        // Store dimensions for debugging
        std::cout << "Starting attention backward pass" << std::endl;
        std::cout << "Hidden size: " << hidden_size << std::endl;
        std::cout << "Num heads: " << num_heads << std::endl;
        std::cout << "Head dim: " << head_dim << std::endl;
        
        // Create local copy of gradient that we can modify
        Matrix grad = grad_output;
        
        // Add gradient norm check with adaptive scaling
        float grad_norm = 0.0f;
        for(size_t i = 0; i < grad.size(); i++) {
            grad_norm += grad.data()[i] * grad.data()[i];
        }
        grad_norm = std::sqrt(grad_norm);
        
        const float MIN_GRAD_NORM = 1e-4f;  // Increased minimum gradient norm
        if (grad_norm < MIN_GRAD_NORM) {
            std::cout << "Warning: Small gradient norm: " << grad_norm << std::endl;
            // Scale up gradients to prevent vanishing
            float scale = MIN_GRAD_NORM / (grad_norm + 1e-8f);
            for(size_t i = 0; i < grad.size(); i++) {
                grad.data()[i] *= scale;
            }
        }

        // Validate dimensions
        validate_dimensions(grad, input, target_distribution);
        
        // Compute gradients with numerical stability
        Matrix dQ = compute_query_gradients(grad, input);
        Matrix dK = compute_key_gradients(grad, input);
        Matrix dV = compute_value_gradients(grad, input);
        
        // Stabilize gradients
        auto stabilize_gradients = [](Matrix& grad) {
            const float MAX_GRAD = 1.0f;
            const float EPSILON = 1e-6f;
            for(size_t i = 0; i < grad.size(); i++) {
                grad.data()[i] = std::clamp(grad.data()[i], -MAX_GRAD, MAX_GRAD);
                if (std::abs(grad.data()[i]) < EPSILON) {
                    grad.data()[i] = grad.data()[i] < 0 ? -EPSILON : EPSILON;
                }
            }
        };
        
        stabilize_gradients(dQ);
        stabilize_gradients(dK);
        stabilize_gradients(dV);

        // Combine gradients
        Matrix combined = combine_gradients(dQ, dK, dV);
        
        return combined;
    } catch (const std::exception& e) {
        std::cerr << "Error in attention backward: " << e.what() << std::endl;
        throw;
    }
}