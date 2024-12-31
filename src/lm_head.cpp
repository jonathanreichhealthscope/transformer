#include "../include/lm_head.hpp"
#include <cmath>

Matrix LanguageModelHead::forward(const Matrix &hidden_states) {
  // Project hidden states to vocabulary size
  // hidden_states is [seq_len x hidden_size]
  // projection is [vocab_size x hidden_size]
  // We want output to be [seq_len x vocab_size]

  // Correct multiplication: [seq_len x hidden_size] * [hidden_size x
  // vocab_size]
  Matrix logits = matmul(hidden_states, projection.transpose());

  // Add bias
  for (size_t i = 0; i < logits.rows(); ++i) {
    for (size_t j = 0; j < logits.cols(); ++j) {
      logits(i, j) += bias[j];
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

Matrix LanguageModelHead::project_to_vocab(const Matrix& hidden_states) const {
    // Project from hidden size to vocabulary size
    Matrix logits(hidden_states.rows(), projection.rows());
    
    // Perform matrix multiplication: hidden_states * weights_^T
    for (size_t i = 0; i < hidden_states.rows(); i++) {
        for (size_t j = 0; j < projection.rows(); j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < projection.cols(); k++) {
                sum += hidden_states(i, k) * projection(j, k);
            }
            logits(i, j) = sum;
        }
    }
    
    return logits;
}