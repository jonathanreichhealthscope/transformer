#include "../../include/optimizer/sam.hpp"
#include <cmath>

float SAM::compute_grad_norm(const std::vector<Matrix>& grads) {
    float total_norm = 0.0f;
    for (const auto& grad : grads) {
        for (size_t i = 0; i < grad.size(); ++i) {
            total_norm += grad.data()[i] * grad.data()[i];
        }
    }
    return std::sqrt(total_norm);
}

void SAM::save_parameter_copies(const std::vector<Matrix*>& params) {
    parameter_copies.clear();
    parameter_copies.reserve(params.size());
    for (const auto* param : params) {
        parameter_copies.push_back(*param);  // Make a copy
    }
}

void SAM::restore_parameters(std::vector<Matrix*>& params) {
    if (params.size() != parameter_copies.size()) {
        throw std::runtime_error("Parameter count mismatch during restore");
    }
    for (size_t i = 0; i < params.size(); ++i) {
        *params[i] = parameter_copies[i];  // Restore the copy
    }
}

void SAM::first_step(std::vector<Matrix*>& params, const std::vector<Matrix>& grads) {
    if (params.size() != grads.size()) {
        throw std::runtime_error("Parameter and gradient count mismatch");
    }

    // Save current parameters
    save_parameter_copies(params);

    // Compute gradient norm
    float grad_norm = compute_grad_norm(grads);
    if (grad_norm == 0.0f) {
        return;
    }

    // Scale factor for gradient
    float scale = rho / (grad_norm + 1e-12f);

    // Update parameters with scaled gradients
    for (size_t i = 0; i < params.size(); ++i) {
        Matrix& param = *params[i];
        const Matrix& grad = grads[i];
        
        for (size_t j = 0; j < param.size(); ++j) {
            param.data()[j] += scale * grad.data()[j];
        }
    }
}

void SAM::second_step(std::vector<Matrix*>& params, const std::vector<Matrix>& grads) {
    // Restore original parameters
    restore_parameters(params);

    // Apply base optimizer update using step() instead of update()
    base_optimizer->step(params, grads);
}

void SAM::update_bias(std::vector<std::reference_wrapper<FloatVector>>& biases,
                     const std::vector<FloatVector>& bias_grads,
                     float learning_rate) {
    if (biases.size() != bias_grads.size()) {
        throw std::runtime_error("Bias and gradient count mismatch");
    }

    // Save current biases
    previous_biases.clear();
    previous_biases.reserve(biases.size());
    for (const auto& bias : biases) {
        previous_biases.push_back(bias.get());
    }

    // Update biases
    for (size_t i = 0; i < biases.size(); ++i) {
        FloatVector& bias = biases[i].get();
        const FloatVector& grad = bias_grads[i];

        if (bias.size() != grad.size()) {
            throw std::runtime_error("Bias and gradient size mismatch");
        }

        // Apply gradient update
        for (size_t j = 0; j < bias.size(); ++j) {
            bias[j] -= learning_rate * grad[j];
        }
    }
}