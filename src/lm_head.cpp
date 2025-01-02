#include "../include/lm_head.hpp"
#include <cmath>
#include <iostream>

Matrix LanguageModelHead::forward_impl(const Matrix &hidden_states) const {
  // Project hidden states to vocabulary size
  // hidden_states: [seq_len × hidden_size] = [5 × 768]
  // projection: [hidden_size × vocab_size] = [768 × 50000]
  // Result: [seq_len × vocab_size] = [5 × 50000]
  std::cout << "In language Model Head: " << std::endl;
  std::cout << "projection: " << projection.rows() << "x" << projection.cols()
            << std::endl;
  Matrix logits = matmul(hidden_states, projection.transpose());

  // Add bias
  for (size_t i = 0; i < logits.rows(); ++i) {
    for (size_t j = 0; j < logits.cols(); ++j) {
      logits(i, j) += bias[j];
    }
  }

  std::cout << "Logits data in language Model Head: " << *logits.data()
            << std::endl;
  return logits;
}

Matrix LanguageModelHead::project_to_vocab(const Matrix &hidden_states) const {
  std::cout << "In project_to_vocab:" << std::endl;

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
  std::cout << "Logits data in project_to_vocab: " << *logits.data() << std::endl;
  return logits;
}

void LanguageModelHead::backward(const Matrix& grad_output, 
                               const Matrix& target_distribution) {
    // Compute cross entropy gradient with respect to logits
    Matrix loss_grad(grad_output.rows(), grad_output.cols());
    for(size_t i = 0; i < grad_output.size(); i++) {
        if (target_distribution.data()[i] > 0.0f) {
            loss_grad.data()[i] = grad_output.data()[i] - target_distribution.data()[i];
        }
    }
    
    // Propagate gradients through the linear layer
    backward_linear(loss_grad);
}

void LanguageModelHead::backward_linear(const Matrix& grad_output) {
    // Compute gradients for projection matrix
    Matrix grad_proj = matmul(grad_output.transpose(), hidden_states);
    
    // Compute gradients for bias
    Vector grad_bias(bias.size(), 0.0f);
    for(size_t i = 0; i < grad_output.rows(); i++) {
        for(size_t j = 0; j < grad_output.cols(); j++) {
            grad_bias[j] += grad_output(i, j);
        }
    }
    
    // Update parameters using gradients
    const float learning_rate = 0.001f;  // You might want to make this configurable
    
    // Update projection matrix
    for(size_t i = 0; i < projection.rows(); i++) {
        for(size_t j = 0; j < projection.cols(); j++) {
            projection(i, j) -= learning_rate * grad_proj(i, j);
        }
    }
    
    // Update bias
    for(size_t i = 0; i < bias.size(); i++) {
        bias[i] -= learning_rate * grad_bias[i];
    }
}