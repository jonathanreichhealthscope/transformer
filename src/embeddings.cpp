#include "embeddings.hpp"
#include <cmath>
#include <random>

TokenEmbedding::TokenEmbedding(size_t vocab_size, size_t hidden_size)
    : vocab_size_(vocab_size), hidden_size_(hidden_size) {
  // Initialize weights with Xavier/Glorot initialization
  std::random_device rd;
  std::mt19937 gen(rd());
  float limit = std::sqrt(6.0f / (vocab_size + hidden_size));
  std::uniform_real_distribution<float> dis(-limit, limit);

  weights = Matrix(vocab_size, hidden_size);
  for (size_t i = 0; i < vocab_size; ++i) {
    for (size_t j = 0; j < hidden_size; ++j) {
      weights(i, j) = dis(gen);
    }
  }
}

Matrix TokenEmbedding::forward(const std::vector<int> &tokens) {
  Matrix output(tokens.size(), hidden_size_);
  for (size_t i = 0; i < tokens.size(); ++i) {
    for (size_t j = 0; j < hidden_size_; ++j) {
      output(i, j) = weights(tokens[i], j);
    }
  }
  return output;
}

Matrix TokenEmbedding::project_to_vocab(const Matrix &hidden_states) {
  // Project back to vocabulary space using transposed weights
  Matrix logits(hidden_states.rows(), vocab_size_);
  for (size_t i = 0; i < hidden_states.rows(); ++i) {
    for (size_t v = 0; v < vocab_size_; ++v) {
      float sum = 0.0f;
      for (size_t h = 0; h < hidden_size_; ++h) {
        sum += hidden_states(i, h) * weights(v, h);
      }
      logits(i, v) = sum;
    }
  }
  return logits;
}

void TokenEmbedding::save(std::ostream &os) const {
  os.write(reinterpret_cast<const char *>(&vocab_size_), sizeof(vocab_size_));
  os.write(reinterpret_cast<const char *>(&hidden_size_), sizeof(hidden_size_));
  os.write(reinterpret_cast<const char *>(weights.data()),
           vocab_size_ * hidden_size_ * sizeof(float));
}

std::unique_ptr<TokenEmbedding> TokenEmbedding::load(std::istream &is) {
  size_t vocab_size, hidden_size;
  is.read(reinterpret_cast<char *>(&vocab_size), sizeof(vocab_size));
  is.read(reinterpret_cast<char *>(&hidden_size), sizeof(hidden_size));

  auto embedding = std::make_unique<TokenEmbedding>(vocab_size, hidden_size);
  is.read(reinterpret_cast<char *>(embedding->weights.data()),
          vocab_size * hidden_size * sizeof(float));
  return embedding;
}

PositionalEncoding::PositionalEncoding(size_t max_seq_length,
                                       size_t hidden_size)
    : encoding_matrix(max_seq_length, hidden_size) {
  // Implement sinusoidal position embeddings
  for (size_t pos = 0; pos < max_seq_length; ++pos) {
    for (size_t i = 0; i < hidden_size; i += 2) {
      float freq = 1.0f / std::pow(10000.0f, (i / float(hidden_size)));
      encoding_matrix(pos, i) = std::sin(pos * freq);
      if (i + 1 < hidden_size) {
        encoding_matrix(pos, i + 1) = std::cos(pos * freq);
      }
    }
  }
}

Matrix PositionalEncoding::forward(const Matrix &position_ids) {
  Matrix output(position_ids.rows(), encoding_matrix.cols());
  for (size_t i = 0; i < position_ids.rows(); ++i) {
    for (size_t j = 0; j < encoding_matrix.cols(); ++j) {
      size_t pos = static_cast<size_t>(position_ids(i, 0));
      output(i, j) = encoding_matrix(pos, j);
    }
  }
  return output;
}

void PositionalEncoding::save(std::ostream &os) const {
  size_t max_seq_length = encoding_matrix.rows();
  size_t hidden_size = encoding_matrix.cols();
  os.write(reinterpret_cast<const char *>(&max_seq_length),
           sizeof(max_seq_length));
  os.write(reinterpret_cast<const char *>(&hidden_size), sizeof(hidden_size));
  os.write(reinterpret_cast<const char *>(encoding_matrix.data()),
           max_seq_length * hidden_size * sizeof(float));
}

std::unique_ptr<PositionalEncoding> PositionalEncoding::load(std::istream &is) {
  size_t max_seq_length, hidden_size;
  is.read(reinterpret_cast<char *>(&max_seq_length), sizeof(max_seq_length));
  is.read(reinterpret_cast<char *>(&hidden_size), sizeof(hidden_size));

  auto pos_encoding =
      std::make_unique<PositionalEncoding>(max_seq_length, hidden_size);
  is.read(reinterpret_cast<char *>(pos_encoding->encoding_matrix.data()),
          max_seq_length * hidden_size * sizeof(float));
  return pos_encoding;
}