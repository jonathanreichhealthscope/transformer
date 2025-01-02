#include "../../include/optimizer/sam.hpp"
#include <algorithm>
#include <cmath>

float SAM::compute_grad_norm(const std::vector<Matrix> &grads) {
  float total_norm = 0.0f;
  const float epsilon = 1e-12f;  // Prevent underflow
  for (const auto &grad : grads) {
    for (size_t i = 0; i < grad.size(); ++i) {
      // Prevent underflow by clamping tiny gradients
      float g = grad.data()[i];
      if (std::abs(g) < epsilon) {
        g = 0.0f;
      }
      total_norm += g * g;
    }
  }
  // Prevent sqrt of zero
  total_norm = std::max(total_norm, epsilon * epsilon);
  return std::sqrt(total_norm);
}

void SAM::save_parameter_copies(const std::vector<Matrix *> &params) {
  parameter_copies.clear();
  parameter_copies.reserve(params.size());
  for (const auto *param : params) {
    parameter_copies.push_back(*param); // Make a copy
  }
}

void SAM::restore_parameters(std::vector<Matrix *> &params) {
  if (params.size() != parameter_copies.size()) {
    throw std::runtime_error("Parameter count mismatch during restore");
  }
  for (size_t i = 0; i < params.size(); ++i) {
    *params[i] = parameter_copies[i]; // Restore the copy
  }
}

void SAM::first_step(std::vector<Matrix *> &params,
                     const std::vector<Matrix> &grads) {
  if (params.size() != grads.size()) {
    throw std::runtime_error("Parameter and gradient count mismatch");
  }

  // Save current parameters
  save_parameter_copies(params);

  // Compute gradient norm
  float grad_norm = compute_grad_norm(grads);
  const float min_norm = 1e-8f;
  if (grad_norm < min_norm) {
    std::cerr << "Warning: Very small gradient norm: " << grad_norm << std::endl;
    return;
  }

  // Scale factor for gradient
  // Clamp scale to prevent extreme values
  float scale = std::min(rho / grad_norm, 10.0f);

  // Update parameters with scaled gradients
  for (size_t i = 0; i < params.size(); ++i) {
    Matrix &param = *params[i];
    const Matrix &grad = grads[i];

    for (size_t j = 0; j < param.size(); ++j) {
      // Prevent parameter updates from becoming too large
      float update = scale * grad.data()[j];
      update = std::clamp(update, -1.0f, 1.0f);
      param.data()[j] += update;
    }
  }
}

void SAM::second_step(std::vector<Matrix *> &params,
                      const std::vector<Matrix> &grads) {
  // Restore original parameters
  restore_parameters(params);

  // Apply base optimizer update using step() instead of update()
  base_optimizer->step(params, grads);
}

void SAM::update_bias(std::vector<std::reference_wrapper<FloatVector>> &biases,
                      const std::vector<FloatVector> &bias_grads,
                      float learning_rate) {
  if (biases.size() != bias_grads.size()) {
    throw std::runtime_error("Bias and gradient count mismatch");
  }

  // Save current biases
  previous_biases.clear();
  previous_biases.reserve(biases.size());
  for (const auto &bias : biases) {
    previous_biases.push_back(bias.get());
  }

  // Update biases
  for (size_t i = 0; i < biases.size(); ++i) {
    FloatVector &bias = biases[i].get();
    const FloatVector &grad = bias_grads[i];

    if (bias.size() != grad.size()) {
      throw std::runtime_error("Bias and gradient size mismatch");
    }

    // Apply gradient update
    for (size_t j = 0; j < bias.size(); ++j) {
      // Clamp gradients and prevent extreme updates
      float grad_value = std::clamp(grad[j], -10.0f, 10.0f);
      float update = learning_rate * grad_value;
      update = std::clamp(update, -0.1f, 0.1f);
      bias[j] -= update;
    }
  }
}