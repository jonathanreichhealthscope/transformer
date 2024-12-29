#pragma once
#include "components.hpp"
#include <memory>
#include <vector>
#include <functional>

class LanguageModelHead {
private:
  Matrix projection;
  Vector bias;
  float dropout_prob;

public:
  LanguageModelHead(size_t hidden_size, size_t vocab_size, float dropout = 0.1)
      : projection(Matrix(vocab_size, hidden_size)), bias(Vector(vocab_size)),
        dropout_prob(dropout) {
    float scale = std::sqrt(1.0f / hidden_size);
    projection.randomize(-scale, scale);
    bias.randomize(-scale, scale);
  }

  Matrix forward(const Matrix &hidden_states);

  Matrix backward(const Matrix &grad_output, const Matrix &hidden_states) {
    // Compute gradients for projection and bias
    Matrix grad_proj = matmul(grad_output, hidden_states);
    Vector grad_bias = grad_output.row_sum();

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
    for (size_t i = 0; i < projection.rows(); ++i) {
      for (size_t j = 0; j < projection.cols(); ++j) {
        // Update momentum
        m_proj(i, j) = beta1 * m_proj(i, j) + (1 - beta1) * grad_proj(i, j);
        // Update RMSprop
        v_proj(i, j) = beta2 * v_proj(i, j) +
                       (1 - beta2) * grad_proj(i, j) * grad_proj(i, j);

        // Bias correction
        float m_hat = m_proj(i, j) / (1 - std::pow(beta1, t));
        float v_hat = v_proj(i, j) / (1 - std::pow(beta2, t));

        // Update weights
        projection(i, j) -= lr * m_hat / (std::sqrt(v_hat) + eps);
      }
    }

    // Update bias vector using Adam optimizer
    for (size_t i = 0; i < bias.size(); ++i) {
      // Update momentum
      m_bias[i] = beta1 * m_bias[i] + (1 - beta1) * grad_bias[i];
      // Update RMSprop
      v_bias[i] = beta2 * v_bias[i] + (1 - beta2) * grad_bias[i] * grad_bias[i];

      // Bias correction
      float m_hat = m_bias[i] / (1 - std::pow(beta1, t));
      float v_hat = v_bias[i] / (1 - std::pow(beta2, t));

      // Update bias
      bias[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }

    // Compute gradient with respect to input
    return matmul(projection.transpose(), grad_output);
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

  Vector& get_bias() { return bias; }
};