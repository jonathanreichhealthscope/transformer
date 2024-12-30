#include "../include/optimizer.hpp"
#include <cmath>

Optimizer::Optimizer(float lr, float b1, float b2, float eps)
    : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

void Optimizer::add_parameter(Matrix &param) {
  parameters.push_back(&param);
  gradients.push_back(Matrix(param.rows(), param.cols(), 0.0f));
}

void Optimizer::update(const std::vector<Matrix> &params,
                       const std::vector<Matrix> &grads) {
  t++;
  for (size_t i = 0; i < parameters.size(); ++i) {
    // Accumulate gradients
    gradients[i] = grads[i];
  }
}

void Optimizer::step() {
  for (size_t i = 0; i < parameters.size(); ++i) {
    // Simple SGD update
    for (size_t j = 0; j < parameters[i]->rows(); ++j) {
      for (size_t k = 0; k < parameters[i]->cols(); ++k) {
        (*parameters[i])(j, k) -= learning_rate * gradients[i](j, k);
      }
    }
  }
  zero_grad();
}

void Optimizer::zero_grad() {
  t = 0; // Reset timestep
}

void Optimizer::save(std::ostream &os) const {
  os.write(reinterpret_cast<const char *>(&learning_rate),
           sizeof(learning_rate));
  os.write(reinterpret_cast<const char *>(&t), sizeof(t));
}

void Optimizer::load(std::istream &is) {
  is.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));
  is.read(reinterpret_cast<char *>(&t), sizeof(t));
}