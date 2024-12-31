#include "../include/lm_head.hpp"
#include <cmath>
#include <iostream>

Matrix LanguageModelHead::forward(const Matrix &hidden_states) const {
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

  return logits;
}

Matrix LanguageModelHead::project_to_vocab(const Matrix& hidden_states) const {
  std::cout << "Debug dimensions in project_to_vocab:" << std::endl;
  std::cout << "hidden_states: " << hidden_states.rows() << "x" << hidden_states.cols() << std::endl;
  std::cout << "projection: " << projection.rows() << "x" << projection.cols() << std::endl;
  std::cout << "projection.transpose(): " << projection.transpose().rows() << "x" 
            << projection.transpose().cols() << std::endl;

  // Project from hidden space [seq_len × hidden_size] to vocab space [seq_len × vocab_size]
  // hidden_states: [5 × 768]
  // projection: [768 × 50000]
  // projection.transpose(): [50000 × 768]
  // Result: [5 × 50000]
  Matrix logits = matmul(hidden_states, projection);
  std::cout << "Logits dimensions after projection: " << logits.rows() << "x" << logits.cols() << std::endl;

  // Add bias
  for (size_t i = 0; i < logits.rows(); ++i) {
    for (size_t j = 0; j < logits.cols(); ++j) {
      logits(i, j) += bias[j];
    }
  }
  
  return logits;
}