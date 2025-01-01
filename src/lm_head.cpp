#include "../include/lm_head.hpp"
#include <cmath>
#include <iostream>

Matrix LanguageModelHead::forward(const Matrix &hidden_states) const {
  // Project hidden states to vocabulary size
  // hidden_states: [seq_len × hidden_size] = [5 × 768]
  // projection: [hidden_size × vocab_size] = [768 × 50000]
  // Result: [seq_len × vocab_size] = [5 × 50000]
  std::cout << "hidden_states: " << hidden_states.rows() << "x" << hidden_states.cols() << std::endl;
  std::cout << "projection: " << projection.rows() << "x" << projection.cols() << std::endl;
  Matrix logits = matmul(hidden_states, projection.transpose());

  // Add bias
  for (size_t i = 0; i < logits.rows(); ++i) {
    for (size_t j = 0; j < logits.cols(); ++j) {
      logits(i, j) += bias[j];
    }
  }

  return logits;
}

Matrix LanguageModelHead::project_to_vocab(const Matrix& hidden_states) const {
  std::cout << "Debug dimensions in project_to_vocab:" << std::endl;
  std::cout << "hidden_states: " << hidden_states.rows() << "x" << hidden_states.cols() << std::endl;
  std::cout << "projection: " << projection.rows() << "x" << projection.cols() << std::endl;

  // Project from hidden space to vocab space
  // hidden_states: [5 × 768]
  // projection: [50000 × 768]
  // projection.transpose(): [768 × 50000]
  // Result: [5 × 50000]
  Matrix logits = matmul(hidden_states, projection.transpose());

  // Add bias
  for (size_t i = 0; i < logits.rows(); ++i) {
    for (size_t j = 0; j < logits.cols(); ++j) {
      logits(i, j) += bias[j];
    }
  }
  
  return logits;
}