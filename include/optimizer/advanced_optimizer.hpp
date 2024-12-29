#pragma once
#include "../components.hpp"
#include "optimizer.hpp"
#include <random>
#include <unordered_map>

class SAM { // Sharpness-Aware Minimization
private:
  float rho;
  std::unique_ptr<Optimizer> base_optimizer;

  float compute_grad_norm(const std::vector<Matrix> &grads) {
    float sum_squares = 0.0f;
    for (const auto &grad : grads) {
      for (size_t i = 0; i < grad.rows(); ++i) {
        for (size_t j = 0; j < grad.cols(); ++j) {
          sum_squares += grad(i, j) * grad(i, j);
        }
      }
    }
    return std::sqrt(sum_squares);
  }

public:
  void first_step(const std::vector<Matrix *> &params,
                  const std::vector<Matrix> &grads) {
    // Compute ε(w) for SAM
    float norm = compute_grad_norm(grads);
    float scale = rho / (norm + 1e-12);

    // First optimization step
    for (size_t i = 0; i < params.size(); i++) {
      *params[i] += scale * grads[i];
    }
  }

  void second_step(const std::vector<Matrix *> &params,
                   const std::vector<Matrix> &grads) {
    // Regular optimization step
    base_optimizer->step(params, grads);
  }
};