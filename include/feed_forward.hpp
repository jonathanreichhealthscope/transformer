#pragma once
#include "components.hpp"
#ifdef USE_CUDA
#include "cuda/cuda_utils.cuh"
#endif

using FloatVector = Vector;

class FeedForward {
private:
  Matrix w1;
  Matrix w2;
  Vector b1;
  Vector b2;
  float dropout_prob;
  Matrix intermediate_cache;

  // Gradient members
  mutable Matrix w1_grad;
  mutable Matrix w2_grad;
  mutable FloatVector b1_grad;
  mutable FloatVector b2_grad;

  // Parameter containers
  struct Parameters {
    std::vector<std::reference_wrapper<Matrix>> matrices;
    std::vector<std::reference_wrapper<Vector>> vectors;
  };

  Parameters params;
  mutable Parameters param_gradients;

public:
  virtual ~FeedForward() = default;
  FeedForward() = default;
  FeedForward(size_t hidden_size, size_t intermediate_size,
              float dropout = 0.1f);
  Matrix forward(const Matrix &x);
  Matrix backward(const Matrix &grad_output, const Matrix &input);
  Matrix backward_cuda(const Matrix &grad, const Matrix &input) const;
  void save(std::ostream &os) const;
  static std::unique_ptr<FeedForward> load(std::istream &is);
  friend class Transformer;

  std::vector<std::reference_wrapper<Matrix>> get_weights() {
    return {std::ref(w1), std::ref(w2)};
  }

  friend class TransformerLayer;

  FloatVector &getBias1() { return b1; }
  FloatVector &getBias2() { return b2; }

  FeedForward(const FeedForward &other)
      : w1(other.w1), w2(other.w2), b1(other.b1), b2(other.b2),
        dropout_prob(other.dropout_prob),
        intermediate_cache(other.intermediate_cache), w1_grad(other.w1_grad),
        w2_grad(other.w2_grad), b1_grad(other.b1_grad), b2_grad(other.b2_grad) {
  }

  FeedForward &operator=(const FeedForward &other) {
    if (this != &other) {
      w1 = other.w1;
      w2 = other.w2;
      b1 = other.b1;
      b2 = other.b2;
      dropout_prob = other.dropout_prob;
      intermediate_cache = other.intermediate_cache;
      w1_grad = other.w1_grad;
      w2_grad = other.w2_grad;
      b1_grad = other.b1_grad;
      b2_grad = other.b2_grad;
    }
    return *this;
  }

  Parameters &parameters() {
    params.matrices.clear();
    params.vectors.clear();

    // Matrix parameters
    params.matrices.emplace_back(w1);
    params.matrices.emplace_back(w2);

    // Vector parameters
    params.vectors.emplace_back(b1);
    params.vectors.emplace_back(b2);

    return params;
  }

  const Parameters &parameter_gradients() const {
    param_gradients.matrices.clear();
    param_gradients.vectors.clear();

    // Matrix gradients
    param_gradients.matrices.emplace_back(
        std::ref(const_cast<Matrix &>(w1_grad)));
    param_gradients.matrices.emplace_back(
        std::ref(const_cast<Matrix &>(w2_grad)));

    // Vector gradients
    param_gradients.vectors.emplace_back(
        std::ref(const_cast<FloatVector &>(b1_grad)));
    param_gradients.vectors.emplace_back(
        std::ref(const_cast<FloatVector &>(b2_grad)));

    return param_gradients;
  }
};