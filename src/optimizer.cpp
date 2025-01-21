#include "../include/optimizer.hpp"
#include <cmath>
#include <omp.h>

Optimizer::Optimizer(float lr, float b1, float b2, float eps)
    : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

void Optimizer::add_parameter(Matrix& param) {
    parameters.push_back(&param);
    gradients.push_back(Matrix(param.rows(), param.cols(), 0.0f));
}

void Optimizer::update(const std::vector<Matrix>& params, const std::vector<Matrix>& grads) {
    t++;
#pragma omp parallel for
    for (size_t i = 0; i < parameters.size(); ++i) {
        // Accumulate gradients
        gradients[i] = grads[i];
    }
}

void Optimizer::step() {
    // Single parallel region for all parameter updates
    for (size_t i = 0; i < parameters.size(); ++i) {
        Matrix& param = *parameters[i];
        const Matrix& grad = gradients[i];

#pragma omp parallel for collapse(2)
        for (size_t j = 0; j < param.rows(); ++j) {
            for (size_t k = 0; k < param.cols(); ++k) {
                param(j, k) -= learning_rate * grad(j, k);
            }
        }
    }
    zero_grad();
}

void Optimizer::zero_grad() {
    t = 0; // Reset timestep
}

void Optimizer::save(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
    os.write(reinterpret_cast<const char*>(&t), sizeof(t));
}

void Optimizer::load(std::istream& is) {
    is.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
    is.read(reinterpret_cast<char*>(&t), sizeof(t));
}