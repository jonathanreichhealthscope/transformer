#pragma once
#include "../components.hpp"
#include <bitset>

class BlockSparseAttention {
private:
  static constexpr size_t BLOCK_SIZE = 32;
  std::vector<std::bitset<1024>> sparsity_mask;
  float density_threshold;

  bool should_compute_block(size_t block_row, size_t block_col) const {
    return sparsity_mask[block_row][block_col];
  }

  void compute_attention_block(const Matrix &Q, const Matrix &K,
                               const Matrix &V, Matrix &output,
                               size_t start_row, size_t start_col);

public:
  BlockSparseAttention(float threshold = 0.1f)
      : density_threshold(threshold), sparsity_mask(1024) {
    // Initialize sparsity pattern
    for (auto &mask : sparsity_mask) {
      for (size_t i = 0; i < 1024; ++i) {
        mask[i] = (float(rand()) / RAND_MAX) < density_threshold;
      }
    }
  }

  Matrix forward(const Matrix &Q, const Matrix &K, const Matrix &V) {
    return compute_block_sparse(Q, K, V);
  }

  Matrix compute_block_sparse(const Matrix &Q, const Matrix &K,
                              const Matrix &V);
};