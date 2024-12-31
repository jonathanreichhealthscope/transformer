#include "../include/attention.hpp"
#include <cmath>
#include <iostream>

Matrix MultiHeadAttention::apply_rope(const Matrix &x, size_t position) const {
  Matrix rotated = x; // Create a copy to store rotated values
  const size_t dim = x.cols();

  // Apply rotary position embedding
  for (size_t i = 0; i < x.rows(); ++i) {
    for (size_t j = 0; j < dim; j += 2) {
      std::cout << "in inner loop for apply_rope" << std::endl;
      float cos_theta = cos_cached(position, j / 2);
      std::cout << "passed cos_cached" << std::endl;
      float sin_theta = sin_cached(position, j / 2);
      std::cout << "passed sin_cached" << std::endl;

      float x1 = x(i, j);
      std::cout << "passed x1" << std::endl;
      float x2 = j + 1 < dim ? x(i, j + 1) : 0.0f;
      std::cout << "passed x2" << std::endl;

      rotated(i, j) = x1 * cos_theta - x2 * sin_theta;
      std::cout << "passed rotated(i, j)" << std::endl;
      if (j + 1 < dim) {
        rotated(i, j + 1) = x1 * sin_theta + x2 * cos_theta;
        std::cout << "passed rotated(i, j + 1)" << std::endl;
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
  std::cout << "Forwarding through MultiHeadAttention" << std::endl;
  // Project input to Q, K, V
  Matrix Q = matmul(x, query_proj);
  std::cout << "Got Q" << std::endl;
  Matrix K = matmul(x, key_proj);
  std::cout << "Got K" << std::endl;
  Matrix V = matmul(x, value_proj);
  std::cout << "Got V" << std::endl;

  // Apply RoPE if enabled
  if (use_rope) {
    std::cout << "Applying RoPE" << std::endl;
    std::cout << "Q rows: " << Q.rows() << std::endl;
    std::cout << "Q cols: " << Q.cols() << std::endl;
    std::cout << "K rows: " << K.rows() << std::endl;
    std::cout << "K cols: " << K.cols() << std::endl;
    for (size_t pos = 0; pos < x.rows(); ++pos) {
      std::cout << "Applying RoPE to position: " << pos << std::endl;
      // Create single-row matrices for RoPE
      Matrix Q_row(1, Q.cols());
      Matrix K_row(1, K.cols());
      for (size_t j = 0; j < Q.cols(); ++j) {
        //std::cout <<  pos << " " << j << " " << Q(pos, j) << std::endl;
        Q_row(0, j) = Q(pos, j);
        //std::cout <<  pos << " " << j << " " << K(pos, j) << std::endl;
        K_row(0, j) = K(pos, j);
      }
      std::cout << "calling apply_rope at position: " << pos << " " << Q_row(0, 0)  << std::endl;
      // Apply RoPE
      Matrix Q_rotated = apply_rope(Q_row, pos);
      std::cout << "calling apply_rope at position: " << pos << " " << K_row(0, 0) << std::endl;
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
    std::cout << "Using standard attention" << std::endl;
    attention_output = standard_attention(Q, K, V, mask);
  }
  std::cout << "Attention output: rows: " << attention_output.rows() << "Attention output: cols: " << attention_output.cols() << std::endl;
  std::cout << "Q rows: " << Q.rows() << std::endl;
  std::cout << "Q cols: " << Q.cols() << std::endl;
  std::cout << "K rows: " << K.rows() << std::endl;
  std::cout << "K cols: " << K.cols() << std::endl;
  std::cout << "V rows: " << V.rows() << std::endl;
  std::cout << "V cols: " << V.cols() << std::endl;
  // Project output
  std::cout << "Projecting output" << std::endl;
  std::cout << "Output proj rows: " << output_proj.rows() << std::endl;
  std::cout << "Output proj cols: " << output_proj.cols() << std::endl;
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

  // Initialize RoPE buffers if needed
  if (use_rope) {
    cos_cached = Matrix(window_size, head_dim / 2);
    sin_cached = Matrix(window_size, head_dim / 2);
    // Initialize RoPE angle cache
    for (size_t pos = 0; pos < window_size; ++pos) {
      for (size_t i = 0; i < head_dim / 2; ++i) {
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
  std::cout << "Standard attention" << std::endl;
  Matrix scores = matmul(Q, K.transpose());
  scores *= 1.0f / std::sqrt(static_cast<float>(head_dim));
  std::cout << "Scores: rows: " << scores.rows() << "Scores: cols: " << scores.cols() << std::endl;
  if (!mask.mask.empty()) {
    for (size_t i = 0; i < scores.rows(); ++i) {
      for (size_t j = 0; j < scores.cols(); ++j) {
        if (mask.mask(i, j) == 0.0f) {
          scores(i, j) = -std::numeric_limits<float>::infinity();
        }
      }
    }
  }

  scores.apply_softmax();
  return matmul(scores, V);
}

Matrix MultiHeadAttention::backward(const Matrix &grad,
                                    const Matrix &input) const {
  // For now, return a simple gradient
  return Matrix(input.rows(), input.cols());
}