#pragma once
#include "components.hpp"
#ifdef USE_CUDA
#include "cuda/cuda_utils.cuh"
#endif

using FloatVector = Vector;

/**
 * @brief Implementation of the Feed-Forward Network (FFN) used in transformer layers.
 * 
 * The Feed-Forward Network consists of two linear transformations with a ReLU activation
 * in between, following the architecture:
 * FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
 * 
 * Features:
 * - Two-layer neural network with ReLU activation
 * - Dropout regularization
 * - CUDA acceleration support
 * - Gradient computation for training
 */
class FeedForward {
  private:
    Matrix w1;                    ///< Weight matrix for the first linear transformation
    Matrix w2;                    ///< Weight matrix for the second linear transformation
    Vector b1;                    ///< Bias vector for the first linear transformation
    Vector b2;                    ///< Bias vector for the second linear transformation
    float dropout_prob;           ///< Dropout probability during training
    Matrix intermediate_cache;    ///< Cache for intermediate activations during forward pass

    // Gradient members
    mutable Matrix w1_grad;       ///< Gradient of loss with respect to w1
    mutable Matrix w2_grad;       ///< Gradient of loss with respect to w2
    mutable FloatVector b1_grad;  ///< Gradient of loss with respect to b1
    mutable FloatVector b2_grad;  ///< Gradient of loss with respect to b2

    /**
     * @brief Container for trainable parameters.
     * 
     * Groups matrices and vectors for easier parameter management
     * and optimization updates.
     */
    struct Parameters {
        std::vector<std::reference_wrapper<Matrix>> matrices;  ///< References to weight matrices
        std::vector<std::reference_wrapper<Vector>> vectors;   ///< References to bias vectors
    };

    Parameters params;                  ///< Container for trainable parameters
    mutable Parameters param_gradients; ///< Container for parameter gradients

  public:
    virtual ~FeedForward() = default;
    FeedForward() = default;

    /**
     * @brief Constructs a feed-forward network with specified dimensions.
     * @param hidden_size Size of input and output tensors
     * @param intermediate_size Size of the intermediate (hidden) layer
     * @param dropout Dropout probability during training
     */
    FeedForward(size_t hidden_size, size_t intermediate_size, float dropout = 0.1f);

    /**
     * @brief Performs the forward pass through the feed-forward network.
     * @param x Input tensor of shape [batch_size, seq_len, hidden_size]
     * @return Output tensor of shape [batch_size, seq_len, hidden_size]
     */
    Matrix forward(const Matrix& x);

    /**
     * @brief Performs the backward pass to compute gradients.
     * @param grad_output Gradient of the loss with respect to the output
     * @param input Original input tensor
     * @return Gradient with respect to the input
     */
    Matrix backward(const Matrix& grad_output, const Matrix& input);

    /**
     * @brief CUDA-accelerated version of the backward pass.
     * @param grad Gradient of the loss with respect to the output
     * @param input Original input tensor
     * @return Gradient with respect to the input
     */
    Matrix backward_cuda(const Matrix& grad, const Matrix& input) const;

    /**
     * @brief Saves the feed-forward network parameters to a stream.
     * @param os Output stream to save to
     */
    void save(std::ostream& os) const;

    /**
     * @brief Loads feed-forward network parameters from a stream.
     * @param is Input stream to load from
     * @return Unique pointer to the loaded feed-forward network
     */
    static std::unique_ptr<FeedForward> load(std::istream& is);

    /**
     * @brief Gets references to all trainable weight matrices.
     * @return Vector of references to weight matrices
     */
    std::vector<std::reference_wrapper<Matrix>> get_weights() {
        return {std::ref(w1), std::ref(w2)};
    }

    friend class Transformer;
    friend class TransformerLayer;

    FloatVector& getBias1() {
        return b1;
    }
    FloatVector& getBias2() {
        return b2;
    }

    FeedForward(const FeedForward& other)
        : w1(other.w1), w2(other.w2), b1(other.b1), b2(other.b2), dropout_prob(other.dropout_prob),
          intermediate_cache(other.intermediate_cache), w1_grad(other.w1_grad),
          w2_grad(other.w2_grad), b1_grad(other.b1_grad), b2_grad(other.b2_grad) {}

    FeedForward& operator=(const FeedForward& other) {
        if (this != &other) {
            w1 = other.w1;
            w2 = other.w2;
            b1 = other.b1;
            b2 = other.b2;
            dropout_prob = other.dropout_prob;
            intermediate_cache = other.intermediate_cache;
            w1_grad = other.w1_grad;
            w2_grad = other.w2_grad;
            b1_grad = other.b1_grad;
            b2_grad = other.b2_grad;
        }
        return *this;
    }

    Parameters& parameters() {
        params.matrices.clear();
        params.vectors.clear();

        // Matrix parameters
        params.matrices.emplace_back(w1);
        params.matrices.emplace_back(w2);

        // Vector parameters
        params.vectors.emplace_back(b1);
        params.vectors.emplace_back(b2);

        return params;
    }

    const Parameters& parameter_gradients() const {
        param_gradients.matrices.clear();
        param_gradients.vectors.clear();

        // Matrix gradients
        param_gradients.matrices.emplace_back(std::ref(const_cast<Matrix&>(w1_grad)));
        param_gradients.matrices.emplace_back(std::ref(const_cast<Matrix&>(w2_grad)));

        // Vector gradients
        param_gradients.vectors.emplace_back(std::ref(const_cast<FloatVector&>(b1_grad)));
        param_gradients.vectors.emplace_back(std::ref(const_cast<FloatVector&>(b2_grad)));

        return param_gradients;
    }
};