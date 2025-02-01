#pragma once
#include "matrix.hpp"


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
public:
    /**
     * @brief Constructs a layer normalization module.
     * @param hidden_size_ Size of the input features
     * @param eps_ Small constant for numerical stability (default: 1e-5)
     */
    LayerNorm(size_t hidden_size_, float eps_ = 1e-5);

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
    Matrix backward(const Matrix& grad_output, const Matrix& input) {
        input_cache_ = input;  // Cache input for backward pass
        return compute_gradients(grad_output);
    }

    /**
     * @brief Gets references to all learnable parameters.
     * @return Vector of references to parameter vectors
     */
    const std::vector<std::reference_wrapper<Matrix>> get_parameter_list() {
        std::vector<std::reference_wrapper<Matrix>> params;
        params.push_back(std::ref(params_.gamma));
        params.push_back(std::ref(params_.beta));
        return params;
    }

    /**
     * @brief Gets references to all parameter gradients.
     * @return Vector of references to gradient vectors
     */
    const std::vector<std::reference_wrapper<const Matrix>> parameter_gradients() const {
        std::vector<std::reference_wrapper<const Matrix>> grads;
        grads.push_back(std::cref(grads_.gamma_grad));
        grads.push_back(std::cref(grads_.beta_grad));
        return grads;
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

    /**
     * @brief Gets the size of the input features.
     * @return Hidden size
     */
    size_t get_hidden_size() const {
        return hidden_size_;
    }

    /**
     * @brief Gets the epsilon value.
     * @return Epsilon constant
     */
    float get_eps() const {
        return eps_;
    }

    /**
     * @brief Copy constructor.
     * @param other LayerNorm instance to copy from
     */
    LayerNorm(const LayerNorm& other)
        : hidden_size_(other.hidden_size_), eps_(other.eps_), params_(other.params_),
          input_cache_(other.input_cache_), output_cache_(other.output_cache_),
          grads_(other.grads_) {}

    /**
     * @brief Assignment operator.
     * @param other LayerNorm instance to assign from
     * @return Reference to this instance
     */
    LayerNorm& operator=(const LayerNorm& other) {
        if (this != &other) {
            hidden_size_ = other.hidden_size_;
            eps_ = other.eps_;
            params_ = other.params_;
            input_cache_ = other.input_cache_;
            output_cache_ = other.output_cache_;
            grads_ = other.grads_;
        }
        return *this;
    }

    Matrix get_combined_gradients() const {
        // Create a matrix that combines both gradients
        Matrix combined(1, hidden_size_ * 2);
        std::copy(grads_.gamma_grad.data(), grads_.gamma_grad.data() + hidden_size_, combined.data());
        std::copy(grads_.beta_grad.data(), grads_.beta_grad.data() + hidden_size_, combined.data() + hidden_size_);
        return combined;
    }

    // Mutable accessors
    Matrix& get_gamma_mut() { return params_.gamma; }
    Matrix& get_beta_mut() { return params_.beta; }

    // Const accessors
    const Matrix& get_gamma() const { return params_.gamma; }
    const Matrix& get_beta() const { return params_.beta; }

    // Parameter structure to hold gamma and beta
    struct Parameters {
        Matrix gamma;  // Scale parameter
        Matrix beta;   // Shift parameter
    };

    // Gradient structure to hold gradients
    struct Gradients {
        Matrix gamma_grad;  // Gradient for gamma
        Matrix beta_grad;   // Gradient for beta
    };

    // Parameter accessors
    Parameters& parameters() { return params_; }
    Gradients& param_gradients() { return grads_; }
    const Parameters& parameters() const { return params_; }
    const Gradients& param_gradients() const { return grads_; }

private:
    size_t hidden_size_;
    float eps_;
    Parameters params_;
    Matrix input_cache_;  // Stored for backward pass
    Matrix output_cache_; // Stored for backward pass
    Gradients grads_;

    // Helper method to compute gradients
    Matrix compute_gradients(const Matrix& grad_output);
};