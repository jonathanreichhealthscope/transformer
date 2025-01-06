#include "../include/layer_norm.hpp"
#include <cmath>
#include <omp.h>

Matrix LayerNorm::forward(const Matrix &input) {
    Matrix output(input.rows(), input.cols());
    normalized = Matrix(input.rows(), input.cols());  // Store normalized values
    
    // Process each sample in batch independently
    #pragma omp parallel for
    for (size_t b = 0; b < input.rows(); b++) {
        // Single pass to compute mean and variance
        float mean = 0.0f, var = 0.0f;
        const float* row = input.data() + b * input.cols();
        for (size_t j = 0; j < input.cols(); j++) {
            mean += row[j];
        }
        mean /= input.cols();
        
        for (size_t j = 0; j < input.cols(); j++) {
            float diff = row[j] - mean;
            var += diff * diff;
        }
        var = std::sqrt(var / input.cols() + eps);

        // Normalize and apply scale/shift
        float* out_row = output.data() + b * input.cols();
        float* norm_row = normalized.data() + b * input.cols();
        for (size_t j = 0; j < input.cols(); j++) {
            norm_row[j] = (row[j] - mean) / var;
            out_row[j] = gamma[j] * norm_row[j] + beta[j];
        }
    }
    return output;
}

Matrix LayerNorm::backward(const Matrix &grad, const Matrix &input) {
  std::cout << "grad shape: " << grad.rows() << "x" << grad.cols() << std::endl;
  std::cout << "input shape: " << input.rows() << "x" << input.cols() << std::endl;
  Matrix dx(grad.rows(), grad.cols());
  std::cout << "dx shape: " << dx.rows() << "x" << dx.cols() << std::endl;
  std::cout << "resetting gradients" << std::endl;
  // Reset gradients
  std::fill(gamma_grad.data(), gamma_grad.data() + gamma_grad.size(), 0.0f);
  std::fill(beta_grad.data(), beta_grad.data() + beta_grad.size(), 0.0f);
  std::cout << "gamma grad shape: " << gamma_grad.size() << std::endl;
  std::cout << "beta grad shape: " << beta_grad.size() << std::endl;
// Compute gradients with respect to normalized inputs
std::cout << "computing gradients from grad" << std::endl;
#pragma omp parallel for
  for (size_t i = 0; i < grad.size(); i++) {
    dx.data()[i] = grad.data()[i] * gamma[i % hidden_size];
  }
  std::cout << "computing gradients for gamma and beta" << std::endl;
  // Compute gradients for gamma and beta
  for (size_t j = 0; j < input.cols(); j++) {
      // Accumulate gradients across batch dimension
      for (size_t b = 0; b < input.rows(); b++) {
          gamma_grad[j] += grad(b, j) * normalized(b, j);
          beta_grad[j] += grad(b, j);
      }
  }
  std::cout << "gamma grad shape: " << gamma_grad.size() << std::endl;
  std::cout << "beta grad shape: " << beta_grad.size() << std::endl;


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