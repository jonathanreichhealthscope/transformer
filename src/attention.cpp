#include "../include/attention.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

Matrix MultiHeadAttention::apply_rope(const Matrix &x, size_t position) const {
  Matrix rotated = x; // Create a copy to store rotated values
  const size_t dim = x.cols();

  // Apply rotary position embedding
  for (size_t i = 0; i < x.rows(); ++i) {
    for (size_t j = 0; j < dim; j += 2) {
      // Add bounds check
      if (j / 2 >= cos_cached.cols()) {
        break;
      }

      float cos_theta = cos_cached(position, j / 2);
      float sin_theta = sin_cached(position, j / 2);
      float x1 = x(i, j);
      float x2 = j + 1 < dim ? x(i, j + 1) : 0.0f;

      rotated(i, j) = x1 * cos_theta - x2 * sin_theta;
      if (j + 1 < dim) {
        rotated(i, j + 1) = x1 * sin_theta + x2 * cos_theta;
      }
    }
  }

  return rotated;
}

Matrix MultiHeadAttention::flash_attention(const Matrix &Q, const Matrix &K,
                                           const Matrix &V,
                                           const AttentionMask &mask) const {

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

  return output;
}

Matrix MultiHeadAttention::forward(const Matrix &x, const AttentionMask &mask,
                                   const std::optional<KVCache> &kv_cache) {
  // Debug input values
  std::cerr << "Input matrix x stats before projection:\n";
  std::cerr << "x size: " << x.size() << ", x(0,0): " << x(0,0) << "\n";
  std::cerr << "x min: " << x.min() << ", x max: " << x.max() << "\n";
  std::cerr << "query_proj size: " << query_proj.size() << ", query_proj(0,0): " << query_proj(0,0) << "\n";

  // Project input to Q, K, V
  Matrix Q = matmul(x, query_proj);
  Matrix K = matmul(x, key_proj);
  Matrix V = matmul(x, value_proj);

  // Validate matrix multiplication results
  if(Q.size() == 0 || K.size() == 0 || V.size() == 0) {
    throw std::runtime_error("Q, K, or V matrices have zero size after projection");
  }

  std::cerr << "After projection:\n";
  std::cerr << "Q size: " << Q.size() << ", Q(0,0): " << Q(0,0) << "\n";
  std::cerr << "K size: " << K.size() << ", K(0,0): " << K(0,0) << "\n";
  std::cerr << "V size: " << V.size() << ", V(0,0): " << V(0,0) << "\n";

  // Add validation
  bool q_all_zero = true;
  bool k_all_zero = true;
  bool v_all_zero = true;

  for(size_t i = 0; i < std::min(size_t(10), Q.size()); i++) {
    if(Q.data()[i] != 0.0f) q_all_zero = false;
    if(K.data()[i] != 0.0f) k_all_zero = false; 
    if(V.data()[i] != 0.0f) v_all_zero = false;
  }

  if(q_all_zero || k_all_zero || v_all_zero) {
    std::cerr << "Warning: Q, K, or V matrices contain all zeros after projection\n";
    std::cerr << "Input matrix stats:\n";
    std::cerr << "x min: " << x.min() << " max: " << x.max() << "\n";
    std::cerr << "Projection matrix stats:\n";
    std::cerr << "query_proj min: " << query_proj.min() << " max: " << query_proj.max() << "\n";
    std::cerr << "key_proj min: " << key_proj.min() << " max: " << key_proj.max() << "\n";
    std::cerr << "value_proj min: " << value_proj.min() << " max: " << value_proj.max() << "\n";
  }

  // Apply RoPE if enabled
  if (use_rope) {
    for (size_t pos = 0; pos < x.rows(); ++pos) {
      // Create single-row matrices for RoPE
      Matrix Q_row(1, Q.cols());
      Matrix K_row(1, K.cols());
      for (size_t j = 0; j < Q.cols(); ++j) {
        Q_row(0, j) = Q(pos, j);
        K_row(0, j) = K(pos, j);
      }
      // Apply RoPE
      Matrix Q_rotated = apply_rope(Q_row, pos);
      Matrix K_rotated = apply_rope(K_row, pos);

      // Copy back
      for (size_t j = 0; j < Q.cols(); ++j) {
        Q(pos, j) = Q_rotated(0, j);
        K(pos, j) = K_rotated(0, j);
      }
    }
  }

  // Use flash attention if enabled
  Matrix attention_output;
  if (use_flash) {
    attention_output = flash_attention(Q, K, V, mask);
  } else {
    attention_output = standard_attention(Q, K, V, mask);
  }
  return matmul(attention_output, output_proj);
}

void MultiHeadAttention::save(std::ostream &os) const {
  // Save dimensions and configuration
  os.write(reinterpret_cast<const char *>(&num_heads), sizeof(num_heads));
  os.write(reinterpret_cast<const char *>(&head_dim), sizeof(head_dim));

  // Save projection matrices
  query_proj.save(os);
  key_proj.save(os);
  value_proj.save(os);
  output_proj.save(os);
}

std::unique_ptr<MultiHeadAttention> MultiHeadAttention::load(std::istream &is) {
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

  return attention;
}

MultiHeadAttention::MultiHeadAttention(size_t hidden_size, size_t num_heads,
                                       size_t head_dim, float dropout_prob,
                                       bool use_flash, bool use_rope,
                                       bool use_sliding_window,
                                       size_t window_size, bool use_gqa,
                                       size_t num_kv_heads)
    : num_heads(num_heads), head_dim(head_dim), use_rope(use_rope),
      use_flash(use_flash), use_sliding_window(use_sliding_window),
      window_size(window_size) {
  // Initialize projection matrices
  query_proj = Matrix(hidden_size, num_heads * head_dim);
  key_proj = Matrix(hidden_size, num_heads * head_dim);
  value_proj = Matrix(hidden_size, num_heads * head_dim);
  output_proj = Matrix(num_heads * head_dim, hidden_size);

  // Use larger initialization scale to prevent tiny values
  float scale = sqrt(6.0f / (hidden_size + head_dim * num_heads));  // He initialization
  query_proj.randomize(-scale, scale);
  key_proj.randomize(-scale, scale);
  value_proj.randomize(-scale, scale);
  output_proj.randomize(-scale, scale);

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
}

Matrix MultiHeadAttention::standard_attention(const Matrix &Q, const Matrix &K,
                                              const Matrix &V,
                                              const AttentionMask &mask) const {
  
  Matrix scores = matmul(Q, K.transpose());
  std::cout << "Q: " << *Q.data() << std::endl;
  std::cout << "K: " << *K.data() << std::endl;
  std::cout << "V: " << *V.data() << std::endl;
  // Add numerical stability to attention scaling
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  scale = std::min(scale, 10.0f);  // Prevent too large scaling
  scores *= scale;

  if (!mask.mask.empty()) {
    for (size_t i = 0; i < scores.rows(); ++i) {
      for (size_t j = 0; j < scores.cols(); ++j) {
        std::cout << "score: " << scores(i, j) << std::endl;
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

  return matmul(scores, V);
}

Matrix MultiHeadAttention::backward(const Matrix &grad,
                                    const Matrix &input) const {
  // For now, return a simple gradient
  return Matrix(input.rows(), input.cols());
}