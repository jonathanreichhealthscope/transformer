#include "../include/layernorm.hpp"
#include <cmath>

Matrix LayerNorm::forward(const Matrix &input) const {
  Matrix output(input.rows(), input.cols());

  for (size_t i = 0; i < input.rows(); ++i) {
    // Compute mean
    float mean = 0.0f;
    for (size_t j = 0; j < input.cols(); ++j) {
      mean += input(i, j);
    }
    mean /= input.cols();

    // Compute variance
    float var = 0.0f;
    for (size_t j = 0; j < input.cols(); ++j) {
      float diff = input(i, j) - mean;
      var += diff * diff;
    }
    var = var / input.cols() + eps_;

    // Normalize and scale
    for (size_t j = 0; j < input.cols(); ++j) {
      output(i, j) =
          ((input(i, j) - mean) / std::sqrt(var)) * gamma_[j] + beta_[j];
    }
  }

  return output;
}

Matrix LayerNorm::backward(const Matrix &grad_output,
                           const Matrix &input) const {
  Matrix grad_input(input.rows(), input.cols());

  for (size_t i = 0; i < input.rows(); ++i) {
    // Compute mean and variance
    float mean = 0.0f;
    float var = 0.0f;
    for (size_t j = 0; j < input.cols(); ++j) {
      mean += input(i, j);
    }
    mean /= input.cols();

    for (size_t j = 0; j < input.cols(); ++j) {
      float diff = input(i, j) - mean;
      var += diff * diff;
    }
    var = var / input.cols() + eps_;

    // Compute gradients
    float inv_std = 1.0f / std::sqrt(var);
    for (size_t j = 0; j < input.cols(); ++j) {
      grad_input(i, j) = grad_output(i, j) * gamma_[j] * inv_std;
    }
  }

  return grad_input;
}

void LayerNorm::save(std::ostream &os) const {
  os.write(reinterpret_cast<const char *>(&eps_), sizeof(eps_));
  gamma_.save(os);
  beta_.save(os);
}

std::unique_ptr<LayerNorm> LayerNorm::load(std::istream &is) {
  float eps;
  is.read(reinterpret_cast<char *>(&eps), sizeof(eps));

  Vector gamma = Vector::load(is);
  Vector beta = Vector::load(is);

  auto ln = std::make_unique<LayerNorm>(gamma.size(), eps);
  ln->gamma_ = std::move(gamma);
  ln->beta_ = std::move(beta);
  return ln;
}