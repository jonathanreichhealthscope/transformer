#include "../../include/optimizer/sam.hpp"
#include <algorithm>
#include <cmath>

float SAM::compute_grad_norm(const std::vector<Matrix>& grads) {
    float total_norm = 0.0f;
    const float epsilon = 1e-12f; // Prevent underflow
    for (const auto& grad : grads) {
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

void SAM::save_parameter_copies(const std::vector<Matrix*>& params) {
    parameter_copies.clear();
    parameter_copies.reserve(params.size());
    for (const auto* param : params) {
        parameter_copies.push_back(*param); // Make a copy
    }
}

void SAM::restore_parameters(std::vector<Matrix*>& params) {
    if (params.size() != parameter_copies.size()) {
        throw std::runtime_error("Parameter count mismatch during restore");
    }
    for (size_t i = 0; i < params.size(); ++i) {
        *params[i] = parameter_copies[i]; // Restore the copy
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
        Matrix& param = *params[i];
        const Matrix& grad = grads[i];

        for (size_t j = 0; j < param.size(); ++j) {
            // Prevent parameter updates from becoming too large
            float update = scale * grad.data()[j];
            update = std::clamp(update, -1.0f, 1.0f);
            param.data()[j] += update;
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
                      const std::vector<FloatVector>& bias_grads, float learning_rate) {
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
            // Clamp gradients and prevent extreme updates
            float grad_value = std::clamp(grad[j], -10.0f, 10.0f);
            float update = learning_rate * grad_value;
            update = std::clamp(update, -0.1f, 0.1f);
            bias[j] -= update;
        }
    }
}

void SAM::compute_parameter_gradients(const Matrix& logits, const Matrix& target_distribution,
                                      std::vector<Matrix>& param_grads) {
    // Initialize gradients
    Matrix loss_grad(logits.rows(), logits.cols());

    // Compute cross entropy gradients with numerical stability
    const float epsilon = 1e-12f;
    for (size_t i = 0; i < logits.size(); i++) {
        if (target_distribution.data()[i] > 0.0f) {
            // Compute stable gradient for cross-entropy loss
            float pred = std::min(std::max(logits.data()[i], epsilon), 1.0f - epsilon);
            loss_grad.data()[i] =
                (pred - target_distribution.data()[i]) / (pred * (1.0f - pred) + epsilon);
        }
    }

    // Backpropagate through network layers
    for (size_t layer = param_grads.size(); layer > 0; --layer) {
        size_t idx = layer - 1;

        // Initialize layer gradients if needed
        if (param_grads[idx].empty()) {
            param_grads[idx] = Matrix(logits.rows(), logits.cols());
        }

        // Compute layer gradients
        for (size_t i = 0; i < param_grads[idx].size(); i++) {
            float grad = loss_grad.data()[i];

            // Apply gradient clipping
            grad = std::clamp(grad, -10.0f, 10.0f);

            // Add small noise for regularization
            float noise = ((float) rand() / RAND_MAX - 0.5f) * 1e-5f;
            grad += noise;

            param_grads[idx].data()[i] = grad;
        }

        // Scale gradients for better training stability
        float scale = 1.0f / std::sqrt(static_cast<float>(layer + 1));
        for (size_t i = 0; i < param_grads[idx].size(); i++) {
            param_grads[idx].data()[i] *= scale;
        }
    }
}

Matrix SAM::compute_gradients(const Matrix& logits, const Matrix& hidden_states,
                              LanguageModelHead* lm_head) {
    // Initialize loss gradient
    Matrix loss_grad(logits.rows(), logits.cols());

    // Compute initial loss gradients with softmax stability
    const float epsilon = 1e-12f;
    for (size_t i = 0; i < logits.rows(); i++) {
        // Find max for numerical stability
        float max_val = logits(i, 0);
        for (size_t j = 1; j < logits.cols(); j++) {
            max_val = std::max(max_val, logits(i, j));
        }

        // Compute stable softmax gradients
        float sum_exp = 0.0f;
        std::vector<float> exp_vals(logits.cols());

        for (size_t j = 0; j < logits.cols(); j++) {
            exp_vals[j] = std::exp(logits(i, j) - max_val);
            sum_exp += exp_vals[j];
        }

        // Compute gradients
        for (size_t j = 0; j < logits.cols(); j++) {
            float softmax_out = exp_vals[j] / (sum_exp + epsilon);
            loss_grad(i, j) =
                softmax_out - (j == 0 ? 1.0f : 0.0f); // Assuming first token is target
        }
    }

    // Backpropagate through language model head
    Matrix grad = lm_head->backward_pass(loss_grad, hidden_states);

    // Apply gradient modifications for stability
    for (size_t i = 0; i < grad.size(); i++) {
        // Gradient clipping
        float g = std::clamp(grad.data()[i], -1.0f, 1.0f);

        // Add gradient noise for regularization
        if (grad.data()[i] != 0.0f) {
            float noise_scale = 1e-4f * std::abs(grad.data()[i]);
            float noise = ((float) rand() / RAND_MAX - 0.5f) * noise_scale;
            g += noise;
        }

        // Apply gradient scaling
        if (std::abs(g) < epsilon) {
            g = 0.0f;
        } else {
            g *= std::min(1.0f / std::abs(g), 10.0f); // Scale large gradients
        }

        grad.data()[i] = g;
    }

    // Store computed gradients for later use
    if (current_gradients.empty() || current_gradients.rows() != grad.rows() ||
        current_gradients.cols() != grad.cols()) {
        current_gradients = Matrix(grad.rows(), grad.cols());
    }
    current_gradients = grad;
    return grad;
}