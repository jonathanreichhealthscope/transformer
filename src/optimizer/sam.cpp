#include "../../include/optimizer/sam.hpp"

float SAM::compute_grad_norm(const std::vector<Matrix> &grads) {
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

void SAM::save_parameter_copies(const std::vector<Matrix *> &params) {
  parameter_copies.clear();
  for (const auto *param : params) {
    parameter_copies.push_back(*param);
  }
}

void SAM::restore_parameters(std::vector<Matrix *> &params) {
  for (size_t i = 0; i < params.size(); ++i) {
    *params[i] = parameter_copies[i];
  }
}

void SAM::first_step(std::vector<Matrix *> &params,
                     const std::vector<Matrix> &grads) {
  save_parameter_copies(params);
  float norm = compute_grad_norm(grads);
  float scale = rho / (norm + 1e-12);

  for (size_t i = 0; i < params.size(); i++) {
    for (size_t r = 0; r < params[i]->rows(); r++) {
      for (size_t c = 0; c < params[i]->cols(); c++) {
        (*params[i])(r, c) += scale * grads[i](r, c);
      }
    }
  }
}

void SAM::second_step(std::vector<Matrix *> &params,
                      const std::vector<Matrix> &grads) {
  restore_parameters(params);
  base_optimizer->step(params, grads);
}