#include "../include/embeddings.hpp"
#include <cmath>
#include <random>

TokenEmbedding::TokenEmbedding(size_t vocab_size, size_t embedding_dim)
    : weights_(vocab_size, embedding_dim), vocab_size_(vocab_size),
      embedding_dim_(embedding_dim) {
  // Initialize weights with Xavier/Glorot initialization
  weights_.randomize(-0.1f, 0.1f);
}

Matrix TokenEmbedding::forward(const std::vector<int> &tokens) {
  Matrix output(tokens.size(), embedding_dim_);
  for (size_t i = 0; i < tokens.size(); ++i) {
    for (size_t j = 0; j < embedding_dim_; ++j) {
      output(i, j) = weights_(tokens[i], j);
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

void TokenEmbedding::backward(const Matrix& grad_output, const std::vector<int>& input_tokens) {
  // For each token in the input sequence
  for (size_t i = 0; i < input_tokens.size(); i++) {
    int token_id = input_tokens[i];
    
    // Update the embedding weights for this token
    for (size_t j = 0; j < embedding_dim_; j++) {
      // Accumulate gradients for the embedding weights
      weights_(token_id, j) -= 0.01f * grad_output(i, j); // Simple SGD update
    }
  }
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

Matrix PositionalEncoding::forward(const Matrix &position_ids) {
  Matrix output(position_ids.rows(), encoding_matrix_.cols());
  for (size_t i = 0; i < position_ids.rows(); ++i) {
    for (size_t j = 0; j < encoding_matrix_.cols(); ++j) {
      size_t pos = static_cast<size_t>(position_ids(i, 0));
      output(i, j) = encoding_matrix_(pos, j);
    }
  }
  return output;
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