#pragma once
#include "../components.hpp"
#include <omp.h>

class SlidingWindowAttention {
private:
  size_t window_size;

  void process_attention_window(const Matrix &Q, const Matrix &K,
                                const Matrix &V, Matrix &output, size_t start,
                                size_t end);

public:
  explicit SlidingWindowAttention(size_t window_size_ = 512)
      : window_size(window_size_) {}

  Matrix compute_local_attention(const Matrix &Q, const Matrix &K,
                                 const Matrix &V);
};