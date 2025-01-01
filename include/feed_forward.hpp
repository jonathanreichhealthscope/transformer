#pragma once
#include "components.hpp"
#ifdef USE_CUDA
#include "cuda/cuda_utils.cuh"
#endif

using FloatVector = Vector;

class FeedForward {
private:
  Matrix w1, w2;
  FloatVector b1, b2;
  float dropout_prob;
  Matrix intermediate_cache;

public:
  virtual ~FeedForward() = default;
  FeedForward() = default;
  FeedForward(size_t hidden_size, size_t intermediate_size, float dropout = 0.1);
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
        dropout_prob(other.dropout_prob) {}

  FeedForward &operator=(const FeedForward &other) {
    if (this != &other) {
      w1 = other.w1;
      w2 = other.w2;
      b1 = other.b1;
      b2 = other.b2;
      dropout_prob = other.dropout_prob;
    }
    return *this;
  }
};