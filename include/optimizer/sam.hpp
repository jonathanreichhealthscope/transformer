#pragma once
#include "optimizer.hpp"
#include <cmath>

class SAM {
private:
  float rho;
  std::unique_ptr<Optimizer> base_optimizer;
  std::vector<Matrix> parameter_copies;

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

  void save_parameter_copies(const std::vector<Matrix *> &params) {
    parameter_copies.clear();
    for (const auto *param : params) {
      parameter_copies.push_back(*param);
    }
  }

  void restore_parameters(std::vector<Matrix *> &params) {
    for (size_t i = 0; i < params.size(); ++i) {
      *params[i] = parameter_copies[i];
    }
  }

public:
  SAM(float rho_ = 0.05f, std::unique_ptr<Optimizer> optimizer = nullptr)
      : rho(rho_), base_optimizer(std::move(optimizer)) {
    if (!base_optimizer) {
      base_optimizer = std::make_unique<AdaFactor>();
    }
  }

  void first_step(std::vector<Matrix *> &params,
                  const std::vector<Matrix> &grads) {
    // Save current parameters
    save_parameter_copies(params);

    // Compute ε(w) for SAM
    float norm = compute_grad_norm(grads);
    float scale = rho / (norm + 1e-12);

    // First optimization step: w + ε(w)
    for (size_t i = 0; i < params.size(); i++) {
      for (size_t r = 0; r < params[i]->rows(); r++) {
        for (size_t c = 0; c < params[i]->cols(); c++) {
          (*params[i])(r, c) += scale * grads[i](r, c);
        }
      }
    }
  }

  void second_step(std::vector<Matrix *> &params,
                   const std::vector<Matrix> &grads) {
    // Restore original parameters
    restore_parameters(params);

    // Regular optimization step
    base_optimizer->step(params, grads);
  }

  void zero_grad() { base_optimizer->zero_grad(); }
};