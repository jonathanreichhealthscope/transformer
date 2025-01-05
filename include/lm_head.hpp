#pragma once
#include "components.hpp"
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class LanguageModelHead {
private:
  Matrix projection;
  Vector bias;
  float dropout_prob;
  size_t vocab_size_;
  size_t hidden_size_;
  Matrix hidden_states;
  void backward_linear(const Matrix &grad_output);
  Matrix forward_impl(const Matrix &hidden_states);

public:
  LanguageModelHead(size_t hidden_size, size_t vocab_size, float dropout = 0.1)
      : projection(Matrix(hidden_size, vocab_size)), bias(Vector(vocab_size)),
        dropout_prob(dropout), vocab_size_(vocab_size),
        hidden_size_(hidden_size) {
    float scale = std::sqrt(1.0f / hidden_size);
    std::cout << "LM Head initialization:" << std::endl;
    std::cout << "Creating projection matrix: [" << hidden_size << " × "
              << vocab_size << "]" << std::endl;
    projection.randomize(-scale, scale);
    bias.randomize(-scale, scale);
  }

  LanguageModelHead(const LanguageModelHead &other)
      : projection(other.projection), bias(other.bias),
        dropout_prob(other.dropout_prob) {}

  LanguageModelHead &operator=(const LanguageModelHead &other) {
    if (this != &other) {
      projection = other.projection;
      bias = other.bias;
      dropout_prob = other.dropout_prob;
    }
    return *this;
  }

  Matrix forward(const Matrix &hidden_states) {
    // Store hidden states for backward pass
    this->hidden_states = hidden_states;
    return project_to_vocab(hidden_states);
  }

  Matrix backward_pass(const Matrix &grad_output, const Matrix &hidden_states) {
    // Compute gradients for projection and bias
    std::cout << "Computing gradients for projection and bias" << std::endl;
    Matrix grad_proj = matmul(grad_output.transpose(), hidden_states);
    std::cout << "grad projection shape: " << grad_proj.shape() << std::endl;
    Vector grad_bias = grad_output.row_sum();
    std::cout << "grad bias size: " << grad_bias.size() << std::endl;

    // Apply weight updates with adaptive learning rate
    float lr = 0.001f;    // Base learning rate
    float beta1 = 0.9f;   // Momentum parameter
    float beta2 = 0.999f; // RMSprop parameter
    float eps = 1e-8f;    // Small constant for numerical stability

    static Matrix m_proj(projection.rows(), projection.cols(),
                         0.0f); // Momentum for projection
    static Matrix v_proj(projection.rows(), projection.cols(),
                         0.0f);              // RMSprop for projection
    static Vector m_bias(bias.size(), 0.0f); // Momentum for bias
    static Vector v_bias(bias.size(), 0.0f); // RMSprop for bias
    static size_t t = 0;                     // Time step
    t++;

    // Update projection matrix using Adam optimizer
    std::cout << "updating projection matrix using Adam optimizer" << std::endl;
    for (size_t i = 0; i < projection.rows(); ++i) {
      for (size_t j = 0; j < projection.cols(); ++j) {
        std::cout << "updating momentum" << std::endl;
        // Update momentum
        m_proj(i, j) = beta1 * m_proj(i, j) + (1 - beta1) * grad_proj(i, j);
        std::cout << "updating RMSprop" << std::endl;
        // Update RMSprop
        v_proj(i, j) = beta2 * v_proj(i, j) +
                       (1 - beta2) * grad_proj(i, j) * grad_proj(i, j);
        std::cout << "calculating bias correction" << std::endl;
        // Bias correction
        float m_hat = m_proj(i, j) / (1 - std::pow(beta1, t));
        float v_hat = v_proj(i, j) / (1 - std::pow(beta2, t));
        std::cout << "updating weights" << std::endl;
        // Update weights
        projection(i, j) -= lr * m_hat / (std::sqrt(v_hat) + eps);
      }
    }

    // Update bias vector using Adam optimizer
    for (size_t i = 0; i < bias.size(); ++i) {
      std::cout << "updating momentum" << std::endl;
      // Update momentum
      m_bias[i] = beta1 * m_bias[i] + (1 - beta1) * grad_bias[i];
      std::cout << "updating RMSprop" << std::endl;
      // Update RMSprop
      v_bias[i] = beta2 * v_bias[i] + (1 - beta2) * grad_bias[i] * grad_bias[i];
      std::cout << "calculating bias correction" << std::endl;
      // Bias correction
      float m_hat = m_bias[i] / (1 - std::pow(beta1, t));
      float v_hat = v_bias[i] / (1 - std::pow(beta2, t));
      std::cout << "updating bias" << std::endl;
      // Update bias
      bias[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
    std::cout << "Gradient with respect to input" << std::endl;
    std::cout << "grad_output dims: " << grad_output.rows() << "x"
              << grad_output.cols() << std::endl;
    std::cout << "projection dims: " << projection.rows() << "x"
              << projection.cols() << std::endl;
    // Compute gradient with respect to input
    Matrix grad_input = matmul(grad_output, projection);
    if (grad_input.cols() != hidden_states.cols()) {
      throw std::runtime_error(
          "Language model head gradient output dimension (" +
          std::to_string(grad_input.cols()) + ") must match hidden size (" +
          std::to_string(hidden_states.cols()) + ")");
    }
    return grad_input;
  }

  void save(std::ostream &os) const {
    projection.save(os);
    bias.save(os);
    os.write(reinterpret_cast<const char *>(&dropout_prob),
             sizeof(dropout_prob));
  }

  static std::unique_ptr<LanguageModelHead> load(std::istream &is) {
    auto lm_head = std::make_unique<LanguageModelHead>(0, 0); // Temporary sizes
    lm_head->projection = Matrix::load(is);
    lm_head->bias = Vector::load(is);
    is.read(reinterpret_cast<char *>(&lm_head->dropout_prob),
            sizeof(lm_head->dropout_prob));
    return lm_head;
  }

  std::vector<std::reference_wrapper<Matrix>> get_parameters() {
    std::vector<std::reference_wrapper<Matrix>> params;
    params.push_back(std::ref(projection));
    // Note: We'll need to handle bias separately since it's a Vector
    return params;
  }

  Vector &get_bias() { return bias; }

  Matrix project_to_vocab(const Matrix &hidden_states);

  const Matrix &get_projection() const { return projection; }

  Matrix backward(const Matrix &grad_output,
                  const Matrix &target_distribution = Matrix());
};