#include "../include/lm_head.hpp"
#include <cmath>

Matrix LanguageModelHead::forward(const Matrix &hidden_states) {
  // Project hidden states to vocabulary size
  // hidden_states is [seq_len x hidden_size]
  // projection is [vocab_size x hidden_size]
  // We want output to be [seq_len x vocab_size]

  // Transpose hidden_states first to match dimensions
  Matrix logits = matmul(projection, hidden_states.transpose());

  // Add bias
  for (size_t i = 0; i < logits.rows(); ++i) {
    for (size_t j = 0; j < logits.cols(); ++j) {
      logits(i, j) += bias[i];
    }
  }

  // Apply softmax
  for (size_t i = 0; i < logits.cols(); ++i) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < logits.rows(); ++j) {
      max_val = std::max(max_val, logits(j, i));
    }

    float sum = 0.0f;
    for (size_t j = 0; j < logits.rows(); ++j) {
      logits(j, i) = std::exp(logits(j, i) - max_val);
      sum += logits(j, i);
    }

    for (size_t j = 0; j < logits.rows(); ++j) {
      logits(j, i) /= sum;
    }
  }

  return logits;
}