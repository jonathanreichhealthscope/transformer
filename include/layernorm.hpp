#pragma once
#include "matrix.hpp"
#include <iostream>
#include <memory>
#include <vector>

// Forward declaration
class LayerNorm;

class LayerNorm {
private:
  Vector gamma_;
  Vector beta_;
  float eps_;

public:
  LayerNorm(size_t hidden_size, float eps = 1e-5)
      : gamma_(hidden_size, 1.0f), beta_(hidden_size, 0.0f), eps_(eps) {}

  // Core functionality
  Matrix forward(const Matrix &input) const;
  Matrix backward(const Matrix &grad_output, const Matrix &input) const;
  Matrix backward_cuda(const Matrix &grad_output, const Matrix &input) const;

  // Accessors
  const Vector &get_gamma() const { return gamma_; }
  const Vector &get_beta() const { return beta_; }
  Vector &get_gamma() { return gamma_; }
  Vector &get_beta() { return beta_; }

  // Serialization
  void save(std::ostream &os) const;
  static std::unique_ptr<LayerNorm> load(std::istream &is);
};