#include "../../include/optimizer/sam.hpp"
#include <string>

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
  std::cout << "SAM second step: Starting parameter restoration\n";

  // First restore the original parameters
  restore_parameters(params);
  std::cout << "SAM second step: Parameters restored to original values\n";

  // Verify dimensions before proceeding
  if (params.size() != grads.size()) {
    throw std::runtime_error("SAM second step: Number of parameters (" +
                             std::to_string(params.size()) +
                             ") doesn't match number of gradients (" +
                             std::to_string(grads.size()) + ")");
  }

  // Verify each parameter-gradient pair dimensions
  for (size_t i = 0; i < params.size(); i++) {
    if (params[i]->rows() != grads[i].rows() ||
        params[i]->cols() != grads[i].cols()) {
      throw std::runtime_error(
          "SAM second step: Dimension mismatch at parameter " +
          std::to_string(i) +
          " - Parameter: " + std::to_string(params[i]->rows()) + "x" +
          std::to_string(params[i]->cols()) +
          ", Gradient: " + std::to_string(grads[i].rows()) + "x" +
          std::to_string(grads[i].cols()));
    }
  }
  std::cout << "SAM second step: All dimensions verified\n";

  // Create copies of gradients for the optimizer
  std::vector<Matrix> optimizer_grads;
  optimizer_grads.reserve(grads.size());
  for (const auto &grad : grads) {
    optimizer_grads.push_back(grad); // Make a copy of each gradient
  }
  std::cout << "SAM second step: Created copies of gradients for optimizer\n";

  // Scale gradients if needed (e.g., by learning rate or other factors)
  float scale_factor = 1.0f; // Adjust this if needed
  for (auto &grad : optimizer_grads) {
    grad *= scale_factor;
  }
  std::cout << "SAM second step: Scaled gradients by factor " << scale_factor
            << "\n";

  // Apply the base optimizer's update
  std::cout << "SAM second step: Applying base optimizer update\n";
  try {
    base_optimizer->step(params, optimizer_grads);
    std::cout
        << "SAM second step: Base optimizer update completed successfully\n";
  } catch (const std::exception &e) {
    throw std::runtime_error(
        std::string("SAM second step: Base optimizer failed: ") + e.what());
  }

  // Verify parameters were actually updated
  bool params_changed = false;
  for (size_t i = 0; i < params.size(); i++) {
    if ((*params[i])(0, 0) != parameter_copies[i](0, 0)) {
      params_changed = true;
      break;
    }
  }

  if (!params_changed) {
    std::cout << "Warning: Parameters might not have been updated\n";
  } else {
    std::cout << "SAM second step: Parameters successfully updated\n";
  }
}