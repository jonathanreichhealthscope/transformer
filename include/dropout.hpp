#pragma once
#include "matrix.hpp"
#include <random>

/**
 * @brief Implements dropout regularization for neural networks.
 * 
 * The Dropout class provides a mechanism for randomly "dropping out" units during training,
 * which helps prevent overfitting. Features include:
 * - Configurable dropout rate
 * - Training/inference mode switching
 * - Automatic scaling during training
 * - Mask caching for backpropagation
 */
class Dropout {
  private:
    float dropout_rate;           ///< Probability of dropping a unit
    mutable std::mt19937 gen{std::random_device{}()}; ///< Random number generator
    mutable Matrix dropout_mask;  ///< Binary mask for dropped units
    mutable bool mask_initialized = false; ///< Whether mask has been initialized

  public:
    /**
     * @brief Constructs a dropout layer.
     * @param rate Probability of dropping each unit (between 0 and 1)
     */
    explicit Dropout(float rate) : dropout_rate(rate) {}

    /**
     * @brief Performs the forward pass with dropout.
     * 
     * During training, randomly drops units with probability dropout_rate and
     * scales remaining units by 1/(1-dropout_rate). During inference, performs
     * no dropout and no scaling.
     * 
     * @param input Input matrix to apply dropout to
     * @param training Whether in training mode (true) or inference mode (false)
     * @return Matrix with dropout applied
     * @throws std::runtime_error if dimensions mismatch between input and mask
     */
    Matrix forward(const Matrix& input, bool training) const {
        if (!training || dropout_rate == 0.0f) {
            return input;
        }

        dropout_mask = Matrix(input.rows(), input.cols());
        std::bernoulli_distribution d(1.0f - dropout_rate);

        for (size_t i = 0; i < dropout_mask.size(); ++i) {
            dropout_mask.data()[i] = d(gen) / (1.0f - dropout_rate);
        }

        mask_initialized = true;

        if (input.rows() != dropout_mask.rows() || input.cols() != dropout_mask.cols()) {
            throw std::runtime_error(
                "Dropout mask dimensions (" + std::to_string(dropout_mask.rows()) + "," +
                std::to_string(dropout_mask.cols()) + ") don't match input dimensions (" +
                std::to_string(input.rows()) + "," + std::to_string(input.cols()) + ")");
        }

        return input.hadamard(dropout_mask);
    }

    /**
     * @brief Performs the backward pass of dropout.
     * 
     * Applies the same dropout mask from the forward pass to the gradient,
     * ensuring consistent gradient flow through the network.
     * 
     * @param grad_output Gradient of the loss with respect to the output
     * @return Gradient with respect to the input
     * @throws std::runtime_error if mask not initialized or dimensions mismatch
     */
    Matrix backward(const Matrix& grad_output) const {
        if (!mask_initialized) {
            throw std::runtime_error(
                "Dropout mask not initialized. Forward pass must be called before backward pass");
        }

        if (grad_output.rows() != dropout_mask.rows() ||
            grad_output.cols() != dropout_mask.cols()) {
            throw std::runtime_error("Gradient dimensions (" + std::to_string(grad_output.rows()) +
                                     "," + std::to_string(grad_output.cols()) +
                                     ") don't match dropout mask dimensions (" +
                                     std::to_string(dropout_mask.rows()) + "," +
                                     std::to_string(dropout_mask.cols()) + ")");
        }

        return grad_output.hadamard(dropout_mask);
    }

    /**
     * @brief Gets the dimensions of the current dropout mask.
     * @return Pair of (rows, columns) representing mask dimensions
     */
    std::pair<size_t, size_t> get_mask_dimensions() const {
        return {dropout_mask.rows(), dropout_mask.cols()};
    }

    void reset_mask() {
        dropout_mask = Matrix();
    }
};