#pragma once
#include "components.hpp"
#include <memory>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

/**
 * @brief Layer Normalization implementation for neural networks.
 * 
 * Layer Normalization normalizes the inputs across the features, applying a learnable
 * scale (gamma) and shift (beta) parameter. The normalization is computed as:
 * y = ((x - mean) / sqrt(variance + eps)) * gamma + beta
 * 
 * Features:
 * - Per-layer normalization
 * - Learnable scale and shift parameters
 * - CUDA acceleration support
 * - Gradient computation for training
 */
class LayerNorm {
  private:
    size_t hidden_size;              ///< Size of the input features
    float eps;                       ///< Small constant for numerical stability
    Vector gamma;                    ///< Scale parameter (learnable)
    Vector beta;                     ///< Shift parameter (learnable)
    Vector mean_cache;               ///< Cached mean values for backward pass
    Vector var_cache;                ///< Cached variance values for backward pass
    Vector norm_cache;               ///< Cached normalized values
    Matrix normalized;               ///< Cached normalized matrix for backward pass

    // Gradient storage
    Vector gamma_grad;               ///< Gradient for gamma parameter
    Vector beta_grad;                ///< Gradient for beta parameter

    // Parameter management
    std::vector<std::reference_wrapper<Vector>> params;      ///< References to learnable parameters
    std::vector<std::reference_wrapper<Vector>> grad_params; ///< References to parameter gradients

  public:
    /**
     * @brief Constructs a layer normalization module.
     * @param hidden_size_ Size of the input features
     * @param eps_ Small constant for numerical stability (default: 1e-5)
     */
    LayerNorm(size_t hidden_size_, float eps_ = 1e-5)
        : hidden_size(hidden_size_), eps(eps_), gamma(hidden_size, 1.0f), beta(hidden_size, 0.0f),
          mean_cache(hidden_size), var_cache(hidden_size), norm_cache(hidden_size),
          gamma_grad(hidden_size), beta_grad(hidden_size), normalized(1, hidden_size) {
        // Initialize parameter references
        params = {std::ref(gamma), std::ref(beta)};
        grad_params = {std::ref(gamma_grad), std::ref(beta_grad)};
    }

    /**
     * @brief Performs the forward pass of layer normalization.
     * @param input Input tensor of shape [batch_size, hidden_size]
     * @return Normalized tensor of the same shape
     */
    Matrix forward(const Matrix& input);

    /**
     * @brief Performs the backward pass to compute gradients.
     * @param grad_output Gradient of the loss with respect to the output
     * @param input Original input tensor
     * @return Gradient with respect to the input
     */
    Matrix backward(const Matrix& grad_output, const Matrix& input);

#ifdef USE_CUDA
    /**
     * @brief CUDA-accelerated version of the backward pass.
     * @param grad_output Gradient of the loss with respect to the output
     * @param input Original input tensor
     * @return Gradient with respect to the input
     */
    Matrix backward_cuda(const Matrix& grad_output, const Matrix& input) const;
#endif

    /**
     * @brief Gets references to all learnable parameters.
     * @return Vector of references to parameter vectors
     */
    std::vector<std::reference_wrapper<Vector>>& parameters() {
        return params;
    }

    /**
     * @brief Gets references to all parameter gradients.
     * @return Vector of references to gradient vectors
     */
    const std::vector<std::reference_wrapper<Vector>>& parameter_gradients() const {
        return grad_params;
    }

    /**
     * @brief Saves the layer parameters to a stream.
     * @param os Output stream to save to
     */
    void save(std::ostream& os) const;

    /**
     * @brief Loads layer parameters from a stream.
     * @param is Input stream to load from
     * @return Unique pointer to the loaded layer
     */
    static std::unique_ptr<LayerNorm> load(std::istream& is);

#ifdef USE_CUDA
    /**
     * @brief Gets the scale parameter vector.
     * @return Const reference to gamma vector
     */
    const Vector& get_gamma() const {
        return gamma;
    }

    /**
     * @brief Gets the shift parameter vector.
     * @return Const reference to beta vector
     */
    const Vector& get_beta() const {
        return beta;
    }

    /**
     * @brief Gets the epsilon value.
     * @return Epsilon constant
     */
    float get_eps() const {
        return eps;
    }

    /**
     * @brief Gets the size of the input features.
     * @return Hidden size
     */
    size_t get_hidden_size() const {
        return hidden_size;
    }
#endif

    /**
     * @brief Copy constructor.
     * @param other LayerNorm instance to copy from
     */
    LayerNorm(const LayerNorm& other)
        : hidden_size(other.hidden_size), eps(other.eps), gamma(other.gamma), beta(other.beta),
          mean_cache(other.mean_cache), var_cache(other.var_cache), norm_cache(other.norm_cache),
          normalized(other.normalized), gamma_grad(other.gamma_grad), beta_grad(other.beta_grad) {
        // Initialize parameter references
        params = {std::ref(gamma), std::ref(beta)};
        grad_params = {std::ref(gamma_grad), std::ref(beta_grad)};
    }

    /**
     * @brief Assignment operator.
     * @param other LayerNorm instance to assign from
     * @return Reference to this instance
     */
    LayerNorm& operator=(const LayerNorm& other) {
        if (this != &other) {
            hidden_size = other.hidden_size;
            eps = other.eps;
            gamma = other.gamma;
            beta = other.beta;
            mean_cache = other.mean_cache;
            var_cache = other.var_cache;
            norm_cache = other.norm_cache;
            normalized = other.normalized;
            gamma_grad = other.gamma_grad;
            beta_grad = other.beta_grad;
            // Update parameter references
            params = {std::ref(gamma), std::ref(beta)};
            grad_params = {std::ref(gamma_grad), std::ref(beta_grad)};
        }
        return *this;
    }
};