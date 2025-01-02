#include "../include/layernorm.hpp"
#include <cmath>

Matrix LayerNorm::forward(const Matrix &x) {
  // Compute mean and variance
  float mean = 0.0f, var = 0.0f;
  for (size_t i = 0; i < x.size(); i++) {
    mean += x.data()[i];
    var += x.data()[i] * x.data()[i];
  }
  mean /= x.size();
  var = var / x.size() - mean * mean;
  float std = sqrt(var + eps_);

  // Store normalized values for backward pass
  normalized = Matrix(x.rows(), x.cols());
  Matrix output(x.rows(), x.cols());
  for (size_t i = 0; i < x.size(); i++) {
    normalized.data()[i] = (x.data()[i] - mean) / std;
    output.data()[i] = gamma_[i % hidden_size_] * normalized.data()[i] + beta_[i % hidden_size_];
  }
  return output;
}

Matrix LayerNorm::backward(const Matrix &grad, const Matrix &input) {
  Matrix dx(grad.rows(), grad.cols());
  
  // Compute gradients with respect to normalized inputs
  for (size_t i = 0; i < grad.size(); i++) {
    dx.data()[i] = grad.data()[i] * gamma_[i % hidden_size_];
  }
  
  return dx;
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