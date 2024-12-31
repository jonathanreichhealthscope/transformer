#include "../include/gradient_checkpoint.hpp"
#include <string>

std::unordered_map<size_t, Matrix> GradientCheckpoint::checkpoints;

void GradientCheckpoint::save_activation(const Matrix &activation,
                                         size_t layer) {
  // Use memory pool for efficient allocation
  Matrix &checkpoint = checkpoints[layer];
  checkpoint = Matrix(activation.rows(), activation.cols());

#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < activation.rows(); ++i) {
    for (size_t j = 0; j < activation.cols(); ++j) {
      checkpoint(i, j) = activation(i, j);
    }
  }
}

Matrix GradientCheckpoint::get_activation(size_t layer) {
  auto it = checkpoints.find(layer);
  if (it == checkpoints.end()) {
    throw std::runtime_error("No checkpoint found for layer " +
                             std::to_string(layer));
  }
  return it->second;
}