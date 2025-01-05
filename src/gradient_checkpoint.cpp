#include "../include/gradient_checkpoint.hpp"
#include <string>

std::unordered_map<size_t, Matrix> GradientCheckpoint::checkpoints;
std::unordered_map<std::string, Matrix> GradientCheckpoint::activation_cache;

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

void GradientCheckpoint::cache_activation(const std::string &key,
                                          const Matrix &activation) {
  activation_cache[key] = Matrix(activation); // Deep copy
}

Matrix GradientCheckpoint::get_activation(const std::string &key) {
  if (!has_activation(key)) {
    throw std::runtime_error("No activation found for key: " + key);
  }
  return activation_cache[key];
}

bool GradientCheckpoint::has_activation(const std::string &key) {
  return activation_cache.find(key) != activation_cache.end();
}

void GradientCheckpoint::clear_cache() {
  activation_cache.clear();
  checkpoints.clear();
}