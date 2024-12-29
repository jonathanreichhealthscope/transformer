#pragma once
#include "optimizer.hpp"
#include <cmath>

class SAM {
private:
  float rho;
  std::unique_ptr<Optimizer> base_optimizer;
  std::vector<Matrix> parameter_copies;

  float compute_grad_norm(const std::vector<Matrix> &grads);
  void save_parameter_copies(const std::vector<Matrix *> &params);
  void restore_parameters(std::vector<Matrix *> &params);

public:
  SAM(float rho_ = 0.05f, std::unique_ptr<Optimizer> optimizer = nullptr)
      : rho(rho_), base_optimizer(std::move(optimizer)) {
    if (!base_optimizer) {
      base_optimizer = std::make_unique<AdaFactor>();
    }
  }

  void first_step(std::vector<Matrix *> &params,
                  const std::vector<Matrix> &grads);
  void second_step(std::vector<Matrix *> &params,
                   const std::vector<Matrix> &grads);
  void zero_grad() { base_optimizer->zero_grad(); }
};