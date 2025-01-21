#pragma once
#include "components.hpp"
#include <vector>

/**
 * @brief Implements the Adam optimizer for neural network training.
 * 
 * The Optimizer class provides an implementation of the Adam optimization
 * algorithm, which combines the benefits of:
 * - Momentum for handling sparse gradients
 * - RMSprop for handling non-stationary objectives
 * - Bias correction for improved early iterations
 * 
 * Reference: https://arxiv.org/abs/1412.6980
 */
class Optimizer {
  private:
    std::vector<Matrix*> parameters;  ///< Pointers to parameter matrices being optimized
    std::vector<Matrix> gradients;    ///< Accumulated gradients for each parameter
    float learning_rate;              ///< Learning rate (α in the paper)
    float beta1;                      ///< Exponential decay rate for momentum (β₁)
    float beta2;                      ///< Exponential decay rate for RMSprop (β₂)
    float epsilon;                    ///< Small constant for numerical stability (ε)
    size_t t;                         ///< Number of timesteps for bias correction

  public:
    /**
     * @brief Constructs an Adam optimizer with specified hyperparameters.
     * 
     * @param lr Learning rate (default: 0.001)
     * @param b1 Beta1 coefficient for momentum (default: 0.9)
     * @param b2 Beta2 coefficient for RMSprop (default: 0.999)
     * @param eps Epsilon for numerical stability (default: 1e-8)
     */
    Optimizer(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f);

    /**
     * @brief Adds a parameter matrix to be optimized.
     * 
     * The optimizer will maintain momentum and RMSprop statistics
     * for each added parameter.
     * 
     * @param param Reference to parameter matrix
     */
    void add_parameter(Matrix& param);

    /**
     * @brief Updates parameters using provided gradients.
     * 
     * Applies the Adam update rule to modify parameters based on
     * their corresponding gradients.
     * 
     * @param params Vector of parameter matrices
     * @param grads Vector of gradient matrices
     * @throws std::runtime_error if params and grads sizes don't match
     */
    void update(const std::vector<Matrix>& params, const std::vector<Matrix>& grads);

    /**
     * @brief Performs one optimization step.
     * 
     * Updates all parameters using their accumulated gradients
     * and the Adam update rule:
     * m_t = β₁m_{t-1} + (1-β₁)g_t
     * v_t = β₂v_{t-1} + (1-β₂)g_t²
     * θ_t = θ_{t-1} - α·m_t/√v_t
     */
    void step();

    /**
     * @brief Zeros out all accumulated gradients.
     * 
     * Should be called before accumulating gradients for
     * the next optimization step.
     */
    void zero_grad();

    /**
     * @brief Saves optimizer state to a stream.
     * 
     * Serializes all optimizer parameters and statistics
     * for later restoration.
     * 
     * @param os Output stream to save to
     */
    void save(std::ostream& os) const;

    /**
     * @brief Loads optimizer state from a stream.
     * 
     * Restores optimizer parameters and statistics from
     * a previously saved state.
     * 
     * @param is Input stream to load from
     */
    void load(std::istream& is);
};