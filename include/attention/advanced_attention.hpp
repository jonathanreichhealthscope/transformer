#pragma once
#include "../components.hpp"
#include <bitset>
#include <omp.h>

class BlockSparseAttention {
private:
  static constexpr size_t BLOCK_SIZE = 32;
  std::vector<std::bitset<1024>> sparsity_mask; // For 1024x1024 attention
  float density_threshold;

  bool should_compute_block(size_t block_row, size_t block_col) const {
    return sparsity_mask[block_row][block_col];
  }

  void compute_attention_block(const Matrix &Q, const Matrix &K,
                               const Matrix &V, Matrix &output,
                               size_t start_row, size_t start_col) {
    const size_t end_row = std::min(start_row + BLOCK_SIZE, Q.rows());
    const size_t end_col = std::min(start_col + BLOCK_SIZE, K.rows());

    for (size_t i = start_row; i < end_row; ++i) {
      for (size_t j = start_col; j < end_col; ++j) {
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

  Matrix compute_block_sparse(const Matrix &Q, const Matrix &K,
                              const Matrix &V) {
    Matrix output(Q.rows(), V.cols());
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < Q.rows(); i += BLOCK_SIZE) {
      for (size_t j = 0; j < K.rows(); j += BLOCK_SIZE) {
        if (should_compute_block(i / BLOCK_SIZE, j / BLOCK_SIZE)) {
          compute_attention_block(Q, K, V, output, i, j);
        }
      }
    }
    return output;
  }
};

class LongformerAttention {
private:
  size_t global_tokens;
  std::vector<size_t> global_token_indices;

  Matrix compute_longformer_attention(const Matrix &Q, const Matrix &K,
                                      const Matrix &V) {
    // Implement Longformer's local + global attention pattern
    return Matrix();
  }
};

class SlidingWindowAttention {
private:
  size_t window_size;

  void process_attention_window(const Matrix &Q, const Matrix &K,
                                const Matrix &V, Matrix &output, size_t start,
                                size_t end) {
    for (size_t i = start; i < end; ++i) {
      for (size_t j = std::max(0UL, i - window_size);
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

public:
  explicit SlidingWindowAttention(size_t window_size_ = 512)
      : window_size(window_size_) {}

  Matrix compute_local_attention(const Matrix &Q, const Matrix &K,
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
};