#include "../../include/attention/block_sparse.hpp"

void BlockSparseAttention::compute_attention_block(
    const Matrix &Q, const Matrix &K, const Matrix &V, Matrix &output,
    size_t start_row, size_t start_col) {

  const size_t end_row = std::min(start_row + BLOCK_SIZE, Q.rows());
  const size_t end_col = std::min(start_col + BLOCK_SIZE, K.rows());

  // Compute attention scores for this block
  for (size_t i = start_row; i < end_row; ++i) {
    for (size_t j = start_col; j < end_col; ++j) {
      float score = 0.0f;
      for (size_t k = 0; k < Q.cols(); ++k) {
        score += Q(i, k) * K(j, k);
      }
      score /= std::sqrt(float(Q.cols()));

      // Apply attention to values
      for (size_t k = 0; k < V.cols(); ++k) {
        output(i, k) += score * V(j, k);
      }
    }
  }
}

Matrix BlockSparseAttention::compute_block_sparse(const Matrix &Q,
                                                  const Matrix &K,
                                                  const Matrix &V) {
  Matrix output(Q.rows(), V.cols(), 0.0f);
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