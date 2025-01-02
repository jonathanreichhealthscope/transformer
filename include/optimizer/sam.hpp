#pragma once
#include "../types.hpp"
#include "optimizer.hpp"
#include "../lm_head.hpp"
#include <memory>

class SAM {
private:
  float rho;
  std::unique_ptr<Optimizer> base_optimizer;
  std::vector<Matrix> parameter_copies;
  std::vector<Matrix> previous_weights;
  std::vector<FloatVector> previous_biases;
  Matrix current_gradients;

  float compute_grad_norm(const std::vector<Matrix> &grads);
  void save_parameter_copies(const std::vector<Matrix *> &params);
  void restore_parameters(std::vector<Matrix *> &params);

public:
  SAM(float rho_ = 0.05f, std::unique_ptr<Optimizer> optimizer = nullptr)
      : rho(rho_) {
    if (!optimizer) {
      base_optimizer = std::make_unique<SGD>();
    } else {
      base_optimizer = std::move(optimizer);
    }
  }

  void compute_parameter_gradients(const Matrix& logits,
                                  const Matrix& target_distribution,
                                  std::vector<Matrix>& param_grads);
  Matrix compute_gradients(const Matrix& logits, 
                          const Matrix& hidden_states,
                          LanguageModelHead* lm_head);
  void first_step(std::vector<Matrix *> &params,
                  const std::vector<Matrix> &grads);
  void second_step(std::vector<Matrix *> &params,
                   const std::vector<Matrix> &grads);
  void zero_grad() { base_optimizer->zero_grad(); }
  void update_bias(std::vector<std::reference_wrapper<FloatVector>> &biases,
                   const std::vector<FloatVector> &bias_grads,
                   float learning_rate = 0.001f);
  const Matrix& get_current_gradients() const { return current_gradients; }
};