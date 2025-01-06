#include "../include/layer_norm.hpp"
#include <cmath>
#include <omp.h>

Matrix LayerNorm::forward(const Matrix &input) {
    Matrix output(input.rows(), input.cols());
    
    // Process each sample in batch independently
    for (size_t b = 0; b < input.rows(); b++) {
        // Calculate mean
        float mean = 0.0f;
        for (size_t j = 0; j < input.cols(); j++) {
            mean += input(b, j);
        }
        mean /= input.cols();
        
        // Calculate variance
        float var = 0.0f;
        for (size_t j = 0; j < input.cols(); j++) {
            float diff = input(b, j) - mean;
            var += diff * diff;
        }
        var /= input.cols();
        
        // Normalize and apply scale/shift
        for (size_t j = 0; j < input.cols(); j++) {
            output(b, j) = gamma[j] * (input(b, j) - mean) / std::sqrt(var + eps) + beta[j];
        }
    }
    return output;
}

Matrix LayerNorm::backward(const Matrix &grad, const Matrix &input) {
  Matrix dx(grad.rows(), grad.cols());

  // Reset gradients
  std::fill(gamma_grad.data(), gamma_grad.data() + gamma_grad.size(), 0.0f);
  std::fill(beta_grad.data(), beta_grad.data() + beta_grad.size(), 0.0f);

// Compute gradients with respect to normalized inputs
#pragma omp parallel for
  for (size_t i = 0; i < grad.size(); i++) {
    dx.data()[i] = grad.data()[i] * gamma[i % hidden_size];
  }

  // Compute gradients for gamma and beta
  for (size_t i = 0; i < grad.size(); i++) {
    size_t param_idx = i % hidden_size;
    gamma_grad[param_idx] += grad.data()[i] * normalized.data()[i];
    beta_grad[param_idx] += grad.data()[i];
  }

  return dx;
}

void LayerNorm::save(std::ostream &os) const {
  os.write(reinterpret_cast<const char *>(&eps), sizeof(eps));
  gamma.save(os);
  beta.save(os);
}

std::unique_ptr<LayerNorm> LayerNorm::load(std::istream &is) {
  float eps;
  is.read(reinterpret_cast<char *>(&eps), sizeof(eps));

  Vector gamma_vec = Vector::load(is);
  Vector beta_vec = Vector::load(is);

  auto ln = std::make_unique<LayerNorm>(gamma_vec.size(), eps);
  ln->gamma = std::move(gamma_vec);
  ln->beta = std::move(beta_vec);
  return ln;
}