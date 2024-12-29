#include "../include/attention.hpp"

AttentionMask AttentionMask::create_causal_mask(size_t size) {
  AttentionMask mask;
  mask.mask = Matrix(size, size, 0.0f);

  // Create lower triangular matrix (1s below diagonal, 0s above)
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      mask.mask(i, j) = 1.0f;
    }
  }

  return mask;
}