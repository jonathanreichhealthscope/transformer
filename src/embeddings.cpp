#include "../include/embeddings.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>

TokenEmbedding::TokenEmbedding(size_t vocab_size, size_t embedding_dim)
    : weights_(vocab_size, embedding_dim),
      weights_grad_(vocab_size, embedding_dim),
      vocab_size_(vocab_size),
      embedding_dim_(embedding_dim) {
  // Initialize weights with scaled random values
  float scale = std::sqrt(0.2f / embedding_dim_);
  weights_.randomize(-scale, scale);
  weights_grad_.fill(0.0f);
  
  // Add validation
  bool all_zero = true;
  for(size_t i = 0; i < std::min(size_t(10), weights_.size()); i++) {
    if(weights_.data()[i] != 0.0f) {
      all_zero = false;
      break;
    }
  }
  if(all_zero) {
    throw std::runtime_error("Embedding weights initialization failed - all values are zero");
  }
}

Matrix TokenEmbedding::forward(const std::vector<int> &tokens) {
  // Input validation
  if (tokens.empty()) {
    throw std::runtime_error("Empty token sequence");
  }
  for (int token : tokens) {
    if (token < 0 || static_cast<size_t>(token) >= vocab_size_) {
      throw std::runtime_error("Token id " + std::to_string(token) + " out of range [0, " + std::to_string(vocab_size_) + ")");
    }
  }

  Matrix output(tokens.size(), embedding_dim_);
  
  // Copy embeddings
  for (size_t i = 0; i < tokens.size(); ++i) {
    for (size_t j = 0; j < embedding_dim_; ++j) {
      float val = weights_(tokens[i], j);
      if (std::isnan(val) || std::isinf(val)) {
        throw std::runtime_error("Invalid embedding value at position (" + 
                               std::to_string(tokens[i]) + "," + 
                               std::to_string(j) + "): " + 
                               std::to_string(val));
      }
      output(i, j) = val;
    }
  }

  // Normalize output embeddings with better numerical stability
  const float eps = 1e-6f;  // Increased epsilon for stability
  for (size_t i = 0; i < tokens.size(); i++) {
    float row_norm = 0.0f;
    // Compute norm for this embedding
    for (size_t j = 0; j < embedding_dim_; j++) {
      float val = output(i, j);
      row_norm += val * val;
    }
    row_norm = std::sqrt(row_norm + eps);
    
    // Clamp the norm to prevent division by very small values
    row_norm = std::max(row_norm, 1e-3f);
    
    // Scale this embedding
    float scale = std::min(1.0f, 1.0f / row_norm);
    for (size_t j = 0; j < embedding_dim_; j++) {
      output(i, j) *= scale;
    }
  }

  // Validate output
  for (size_t i = 0; i < output.size(); i++) {
    if (std::isnan(output.data()[i]) || std::isinf(output.data()[i])) {
      throw std::runtime_error("Invalid value in output embeddings at position " + 
                             std::to_string(i));
    }
  }

  return output;
}

Matrix TokenEmbedding::project_to_vocab(const Matrix &hidden_states) {
  Matrix logits(hidden_states.rows(), vocab_size_);
  for (size_t i = 0; i < hidden_states.rows(); ++i) {
    for (size_t v = 0; v < vocab_size_; ++v) {
      float sum = 0.0f;
      for (size_t h = 0; h < embedding_dim_; ++h) {
        sum += hidden_states(i, h) * weights_(v, h);
      }
      // Prevent extreme values in logits
      sum = std::clamp(sum, -100.0f, 100.0f);
      logits(i, v) = sum;
    }
  }
  return logits;
}

void TokenEmbedding::save(std::ostream &os) const {
  os.write(reinterpret_cast<const char *>(&vocab_size_), sizeof(vocab_size_));
  os.write(reinterpret_cast<const char *>(&embedding_dim_),
           sizeof(embedding_dim_));
  weights_.save(os);
}

std::unique_ptr<TokenEmbedding> TokenEmbedding::load(std::istream &is) {
  size_t vocab_size, embedding_dim;
  is.read(reinterpret_cast<char *>(&vocab_size), sizeof(vocab_size));
  is.read(reinterpret_cast<char *>(&embedding_dim), sizeof(embedding_dim));

  auto embedding = std::make_unique<TokenEmbedding>(vocab_size, embedding_dim);
  embedding->weights_ = Matrix::load(is);
  return embedding;
}

void TokenEmbedding::backward(const Matrix &grad_output,
                              const std::vector<int> &input_tokens) {
  // Debug dimension information
  std::cout << "Gradient output dimensions: " << grad_output.rows() << "x"
            << grad_output.cols() << "\n";
  std::cout << "Embedding weights dimensions: " << weights_.rows() << "x"
            << weights_.cols() << "\n";
  std::cout << "Input tokens size: " << input_tokens.size() << "\n";

  // Verify dimensions
  if (grad_output.cols() != embedding_dim_) {
    throw std::runtime_error("Gradient output dimension (" +
                             std::to_string(grad_output.cols()) +
                             ") must match embedding dimension (" +
                             std::to_string(embedding_dim_) + ")");
  }

  // Add check for input_tokens size matching grad_output rows
  if (input_tokens.size() != grad_output.rows()) {
    std::cout << "Warning: Input tokens size (" << input_tokens.size() 
              << ") doesn't match gradient rows (" << grad_output.rows() 
              << "). Using minimum of the two." << std::endl;
  }

  // Initialize gradient accumulator matrix with same dimensions as weights
  Matrix weight_grads(weights_.rows(), weights_.cols(), 0.0f);
  std::cout << "Weight grads dimensions: " << weight_grads.shape() << std::endl;

  // Use the minimum of input_tokens size and grad_output rows to prevent out of bounds
  size_t seq_length = std::min(input_tokens.size(), grad_output.rows());

  // For each token in the input sequence
  for (size_t i = 0; i < seq_length; i++) {
    int token_id = input_tokens[i];
    if (token_id >= static_cast<int>(weights_.rows())) {
      throw std::runtime_error("Token ID " + std::to_string(token_id) +
                               " exceeds vocabulary size " +
                               std::to_string(weights_.rows()));
    }

    // Accumulate gradients
    for (size_t j = 0; j < embedding_dim_; j++) {
      weight_grads(token_id, j) += grad_output(i, j);
    }
  }

  // Apply gradients with learning rate
  const float learning_rate = 0.01f;
  std::cout << "Applying gradients with dimensions check...\n";
  for (size_t i = 0; i < weights_.rows(); i++) {
    for (size_t j = 0; j < weights_.cols(); j++) {
      weights_(i, j) -= learning_rate * weight_grads(i, j);
      if (weight_grads(i, j) != 0.0f) {
        std::cout << "Non-zero gradient at position (" << i << "," << j
                  << "): " << weight_grads(i, j) << "\n";
      }
    }

  }
  std::cout << "Gradient application complete\n";
}

PositionalEncoding::PositionalEncoding(size_t max_seq_length,
                                       size_t hidden_size)
    : encoding_matrix_(max_seq_length, hidden_size),
      max_seq_length_(max_seq_length), hidden_size_(hidden_size) {
  // Implement sinusoidal position embeddings
  for (size_t pos = 0; pos < max_seq_length; ++pos) {
    for (size_t i = 0; i < hidden_size; i += 2) {
      float freq = 1.0f / std::pow(10000.0f, (i / float(hidden_size)));
      encoding_matrix_(pos, i) = std::sin(pos * freq);
      if (i + 1 < hidden_size) {
        encoding_matrix_(pos, i + 1) = std::cos(pos * freq);
      }
    }
  }
}

Matrix PositionalEncoding::forward(const Matrix& position_ids) {
    size_t seq_length = position_ids.rows();
    Matrix encodings(seq_length, hidden_size_);
    
    // Generate positional encodings
    for (size_t pos = 0; pos < seq_length; ++pos) {
        for (size_t i = 0; i < hidden_size_; i += 2) {
            float angle = position_ids(pos, 0) / std::pow(10000.0f, (2.0f * i) / hidden_size_);
            encodings(pos, i) = std::sin(angle);
            if (i + 1 < hidden_size_) {
                encodings(pos, i + 1) = std::cos(angle);
            }
        }
    }
    
    return encodings;
}

void PositionalEncoding::save(std::ostream &os) const {
  size_t max_seq_length = encoding_matrix_.rows();
  size_t hidden_size = encoding_matrix_.cols();
  os.write(reinterpret_cast<const char *>(&max_seq_length),
           sizeof(max_seq_length));
  os.write(reinterpret_cast<const char *>(&hidden_size), sizeof(hidden_size));
  encoding_matrix_.save(os);
}

std::unique_ptr<PositionalEncoding> PositionalEncoding::load(std::istream &is) {
  size_t max_seq_length, hidden_size;
  is.read(reinterpret_cast<char *>(&max_seq_length), sizeof(max_seq_length));
  is.read(reinterpret_cast<char *>(&hidden_size), sizeof(hidden_size));

  auto pos_encoding =
      std::make_unique<PositionalEncoding>(max_seq_length, hidden_size);
  pos_encoding->encoding_matrix_ = Matrix::load(is);
  return pos_encoding;
}