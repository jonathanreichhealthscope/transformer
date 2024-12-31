#pragma once
#include "components.hpp"

class LayerNorm {
private:
  Vector gamma;
  Vector beta;
  float eps;

public:
  virtual ~LayerNorm() = default;
  LayerNorm() : eps(1e-5) {}
  LayerNorm(size_t hidden_size, float eps = 1e-5);
  Matrix forward(const Matrix &x) const;
  Matrix forward_cuda(const Matrix &x) const;
  void save(std::ostream &os) const;
  static std::unique_ptr<LayerNorm> load(std::istream &is);
  Matrix backward(const Matrix &grad, const Matrix &input) const;
  Matrix backward_cuda(const Matrix &grad, const Matrix &input) const;
  friend class Transformer;

  LayerNorm(const LayerNorm& other)
      : gamma(other.gamma), beta(other.beta), eps(other.eps) {}
  
  LayerNorm& operator=(const LayerNorm& other) {
      if (this != &other) {
          gamma = other.gamma;
          beta = other.beta;
          eps = other.eps;
      }
      return *this;
  }
};