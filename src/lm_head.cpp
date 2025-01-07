#include "../include/lm_head.hpp"
#include <cmath>
#include <iostream>

LanguageModelHead::LanguageModelHead(size_t hidden_size, size_t vocab_size)
    : hidden_size_(hidden_size), 
      vocab_size_(vocab_size),
      projection(hidden_size, vocab_size),
      bias(vocab_size, 0.0f),
      token_frequencies(vocab_size, 0.0f)  // Initialize frequencies
{
    float scale = std::sqrt(1.0f / hidden_size);
    std::cout << "LM Head initialization:" << std::endl;
    std::cout << "Creating projection matrix: [" << hidden_size << " × "
              << vocab_size << "]" << std::endl;
    projection.randomize(-scale, scale);
    bias.randomize(-scale, scale);
}

Matrix LanguageModelHead::forward_impl(const Matrix &hidden_states) {
  // Store hidden states for backward pass
  this->hidden_states = hidden_states;
  
  std::cout << "In language Model Head: " << std::endl;
  std::cout << "Hidden states dimensions: " << hidden_states.rows() << "x" << hidden_states.cols() << std::endl;
  std::cout << "Projection dimensions: " << projection.rows() << "x" << projection.cols() << std::endl;

  // Check dimensions before multiplication
  if (hidden_states.cols() != projection.rows()) {
      throw std::runtime_error("Invalid matrix dimensions for projection: hidden_states.cols() (" + 
          std::to_string(hidden_states.cols()) + ") must match projection.rows() (" + 
          std::to_string(projection.rows()) + ")");
  }

  // Project hidden states to vocabulary size
  // [batch_size x hidden_size] * [hidden_size x vocab_size] = [batch_size x vocab_size]
  Matrix logits = matmul(hidden_states, projection);

  // Add bias
  for (size_t i = 0; i < logits.rows(); ++i) {
    for (size_t j = 0; j < logits.cols(); ++j) {
      logits(i, j) += bias[j];
    }
  }
  return logits;
}

Matrix LanguageModelHead::project_to_vocab(const Matrix &hidden_states) {
  // Store hidden states for backward pass
  this->hidden_states = hidden_states;
  
  std::cout << "In project_to_vocab:" << std::endl;
  std::cout << "Hidden states dimensions: " << hidden_states.rows() << "x" << hidden_states.cols() << std::endl;
  std::cout << "Projection dimensions: " << projection.rows() << "x" << projection.cols() << std::endl;

  // Check dimensions before multiplication
  if (hidden_states.cols() != projection.rows()) {
      throw std::runtime_error("Invalid matrix dimensions for projection: hidden_states.cols() (" + 
          std::to_string(hidden_states.cols()) + ") must match projection.rows() (" + 
          std::to_string(projection.rows()) + ")");
  }

  // Project from hidden space to vocab space
  // [batch_size x hidden_size] * [hidden_size x vocab_size] = [batch_size x vocab_size]
  // Ensure projection matrix has correct dimensions
  if (projection.rows() != hidden_size_ || projection.cols() != vocab_size_) {
      throw std::runtime_error("Projection matrix has wrong dimensions: expected [" + 
          std::to_string(hidden_size_) + " x " + std::to_string(vocab_size_) + 
          "], got [" + std::to_string(projection.rows()) + " x " + 
          std::to_string(projection.cols()) + "]");
  }
  
  Matrix logits = matmul(hidden_states, projection);

  // Apply temperature scaling before vocab balancing
  const float temperature = 0.7f;  // Lower temperature = sharper predictions
  for (size_t i = 0; i < logits.rows(); i++) {
      for (size_t j = 0; j < logits.cols(); j++) {
          logits(i, j) /= temperature;
      }
  }

  // Adjust vocabulary balancing parameters
  const float freq_penalty = 0.15f;  // Increased from 0.1f for stronger effect
  for (size_t i = 0; i < logits.rows(); i++) {
      for (size_t j = 0; j < logits.cols(); j++) {
          // Penalize frequently occurring tokens with logarithmic scaling
          if (token_frequencies[j] > 0) {
              float penalty = freq_penalty * std::log(1 + token_frequencies[j]);
              logits(i,j) -= penalty;
          }
      }
  }
  
  // Add bias
  for (size_t i = 0; i < logits.rows(); ++i) {
    for (size_t j = 0; j < logits.cols(); ++j) {
      logits(i, j) += bias[j];
    }
  }
  return logits;
}

Matrix LanguageModelHead::backward(const Matrix& grad_output, 
                                  const Matrix& target_distribution) {
    std::cout << "LM Head backward dimensions:" << std::endl;
    std::cout << "grad_output: " << grad_output.rows() << "x" << grad_output.cols() << std::endl;
    std::cout << "projection: " << projection.rows() << "x" << projection.cols() << std::endl;
    std::cout << "hidden_states: " << hidden_states.rows() << "x" << hidden_states.cols() << std::endl;
    
    // Compute cross entropy gradient with respect to logits
    Matrix loss_grad(grad_output.rows(), grad_output.cols());
    
    if (!target_distribution.empty()) {
        std::cout << "target_distribution: " << target_distribution.rows() << "x" << target_distribution.cols() << std::endl;
        // If target distribution is provided, compute cross entropy gradient
        for(size_t i = 0; i < grad_output.rows(); i++) {
            for(size_t j = 0; j < grad_output.cols(); j++) {
                size_t idx = i * grad_output.cols() + j;
                if (target_distribution.data()[i] > 0.0f) {
                    loss_grad(i, j) = grad_output(i, j) - target_distribution(i, j);
                }
            }
        }
    } else {
        // Otherwise, just use the provided gradients
        loss_grad = grad_output;
    }
    
    // Propagate gradients through the linear layer
    backward_linear(loss_grad);
    
    // Return gradients with respect to hidden states
    // loss_grad: [batch_size x vocab_size], projection: [hidden_size x vocab_size]
    // Need to transpose projection to get [vocab_size x hidden_size]
    return matmul(loss_grad, projection.transpose());
}

void LanguageModelHead::backward_linear(const Matrix& grad_output) {
    // Check dimensions before matrix multiplication
    if (grad_output.rows() != hidden_states.rows()) {
        throw std::runtime_error("Invalid matrix dimensions for gradient computation: " + 
            std::to_string(grad_output.rows()) + " != " + 
            std::to_string(hidden_states.rows()));
    }
    
    // Compute gradients for projection matrix
    // hidden_states: [batch_size x hidden_size], grad_output: [batch_size x vocab_size]
    // Result should be [hidden_size x vocab_size]
    Matrix grad_proj = matmul(hidden_states.transpose(), grad_output);
    
    // Verify gradient dimensions
    if (grad_proj.rows() != projection.rows() || grad_proj.cols() != projection.cols()) {
        throw std::runtime_error("Gradient dimensions don't match projection matrix: " +
            std::to_string(grad_proj.rows()) + "x" + std::to_string(grad_proj.cols()) +
            " vs " + std::to_string(projection.rows()) + "x" + std::to_string(projection.cols()));
    }
    
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