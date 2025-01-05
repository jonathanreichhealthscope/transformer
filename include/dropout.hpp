#pragma once
#include "matrix.hpp"
#include <random>

class Dropout {
private:
  float dropout_rate;
  mutable std::mt19937 gen{std::random_device{}()};
  mutable Matrix dropout_mask;
  mutable bool mask_initialized = false;

public:
  explicit Dropout(float rate) : dropout_rate(rate) {}

  Matrix forward(const Matrix &input, bool training) const {
    if (!training || dropout_rate == 0.0f) {
      return input;
    }

    dropout_mask = Matrix(input.rows(), input.cols());
    std::bernoulli_distribution d(1.0f - dropout_rate);

    for (size_t i = 0; i < dropout_mask.size(); ++i) {
      dropout_mask.data()[i] = d(gen) / (1.0f - dropout_rate);
    }

    mask_initialized = true;

    if (input.rows() != dropout_mask.rows() ||
        input.cols() != dropout_mask.cols()) {
      throw std::runtime_error(
          "Dropout mask dimensions (" + std::to_string(dropout_mask.rows()) +
          "," + std::to_string(dropout_mask.cols()) +
          ") don't match input dimensions (" + std::to_string(input.rows()) +
          "," + std::to_string(input.cols()) + ")");
    }

    return input.hadamard(dropout_mask);
  }

  Matrix backward(const Matrix &grad_output) const {
    if (!mask_initialized) {
      throw std::runtime_error("Dropout mask not initialized. Forward pass "
                               "must be called before backward pass");
    }

    if (grad_output.rows() != dropout_mask.rows() ||
        grad_output.cols() != dropout_mask.cols()) {
      throw std::runtime_error("Gradient dimensions (" +
                               std::to_string(grad_output.rows()) + "," +
                               std::to_string(grad_output.cols()) +
                               ") don't match dropout mask dimensions (" +
                               std::to_string(dropout_mask.rows()) + "," +
                               std::to_string(dropout_mask.cols()) + ")");
    }

    return grad_output.hadamard(dropout_mask);
  }

  std::pair<size_t, size_t> get_mask_dimensions() const {
    return {dropout_mask.rows(), dropout_mask.cols()};
  }
};