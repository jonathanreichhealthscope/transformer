#include "../include/attention.hpp"

void SlidingWindowAttention::process_attention_window(
    const Matrix &Q, const Matrix &K, const Matrix &V, Matrix &output,
    size_t start, size_t end) {
  for (size_t i = start; i < end; ++i) {
    for (size_t j = std::max<size_t>(0UL, i - window_size);
         j < std::min(K.rows(), i + window_size); ++j) {
      float score = 0.0f;
      for (size_t k = 0; k < Q.cols(); ++k) {
        score += Q(i, k) * K(j, k);
      }
      score /= std::sqrt(float(Q.cols()));

      for (size_t k = 0; k < V.cols(); ++k) {
        output(i, k) += score * V(j, k);
      }
    }
  }
}

Matrix SlidingWindowAttention::compute_local_attention(const Matrix &Q,
                                                       const Matrix &K,
                                                       const Matrix &V) {
  Matrix output(Q.rows(), V.cols(), 0.0f);
  const size_t num_windows = (Q.rows() + window_size - 1) / window_size;

#pragma omp parallel for
  for (size_t w = 0; w < num_windows; w++) {
    size_t start = w * window_size;
    size_t end = std::min(start + window_size, Q.rows());
    process_attention_window(Q, K, V, output, start, end);
  }
  return output;
}