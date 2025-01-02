#pragma once
#include "matrix.hpp"
#include <iostream>
#include <memory>
#include <vector>

// Forward declaration
class LayerNorm;

class LayerNorm {
private:
  size_t hidden_size_;
  float eps_;
  Vector gamma_;  // Scale parameter (keep as Vector for CUDA compatibility)
  Vector beta_;   // Shift parameter
  Matrix normalized;  // Store normalized values for backward pass

public:
  LayerNorm(size_t hidden_size, float eps = 1e-5)
      : hidden_size_(hidden_size), eps_(eps),
        gamma_(hidden_size, 1.0f), beta_(hidden_size, 0.0f) {}
  Matrix forward(const Matrix& x);
  Matrix backward(const Matrix& grad, const Matrix& input);
  Matrix backward_cuda(const Matrix &grad_output, const Matrix &input) const;
  const Matrix& get_normalized() const { return normalized; }
  const Vector& gamma() const { return gamma_; }
  const Vector& beta() const { return beta_; }
  float eps() const { return eps_; }
  size_t hidden_size() const { return hidden_size_; }
  void save(std::ostream& os) const;
  static std::unique_ptr<LayerNorm> load(std::istream& is);
  const Vector& get_gamma() const { return gamma_; }
  const Vector& get_beta() const { return beta_; }
  Vector& get_gamma() { return gamma_; }
  Vector& get_beta() { return beta_; }
};