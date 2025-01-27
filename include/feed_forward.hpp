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
    Matrix input_cache_;         ///< Cache for input during backward pass
    Matrix dropout_mask_;        ///< Dropout mask for training
    
    // Gradient members
    Matrix dW1_;                 ///< Gradient of loss with respect to w1
    Matrix dW2_;                 ///< Gradient of loss with respect to w2
    Vector db1_;                 ///< Gradient of loss with respect to b1
    Vector db2_;                 ///< Gradient of loss with respect to b2

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

    // Training mode control
    bool training_ = true;

  public:
    virtual ~FeedForward() = default;
    FeedForward() = default;

    /**
     * @brief Constructs a feed-forward network with specified dimensions.
     * @param input_size Size of input tensors
     * @param hidden_size Size of the intermediate (hidden) layer
     * @param dropout Dropout probability during training
     */
    FeedForward(size_t input_size, size_t hidden_size, float dropout = 0.1f);

    /**
     * @brief Performs the forward pass through the feed-forward network.
     * @param input Input tensor of shape [batch_size, seq_len, input_size]
     * @return Output tensor of shape [batch_size, seq_len, hidden_size]
     */
    Matrix forward(const Matrix& input);

    /**
     * @brief Performs the backward pass to compute gradients.
     * @param grad_output Gradient of the loss with respect to the output
     * @param input Original input tensor
     * @return Gradient with respect to the input
     */
    Matrix backward(const Matrix& grad_output, const Matrix& input);

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
          intermediate_cache(other.intermediate_cache), input_cache_(other.input_cache_),
          dropout_mask_(other.dropout_mask_), dW1_(other.dW1_), dW2_(other.dW2_),
          db1_(other.db1_), db2_(other.db2_) {}

    FeedForward& operator=(const FeedForward& other) {
        if (this != &other) {
            w1 = other.w1;
            w2 = other.w2;
            b1 = other.b1;
            b2 = other.b2;
            dropout_prob = other.dropout_prob;
            intermediate_cache = other.intermediate_cache;
            input_cache_ = other.input_cache_;
            dropout_mask_ = other.dropout_mask_;
            dW1_ = other.dW1_;
            dW2_ = other.dW2_;
            db1_ = other.db1_;
            db2_ = other.db2_;
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
        param_gradients.matrices.emplace_back(std::ref(const_cast<Matrix&>(dW1_)));
        param_gradients.matrices.emplace_back(std::ref(const_cast<Matrix&>(dW2_)));

        // Vector gradients
        param_gradients.vectors.emplace_back(std::ref(const_cast<Vector&>(db1_)));
        param_gradients.vectors.emplace_back(std::ref(const_cast<Vector&>(db2_)));

        return param_gradients;
    }

    // Training mode control
    void set_training(bool training_mode) { training_ = training_mode; }
    bool is_training() const { return training_; }

    // Parameter updates
    void update_parameters(const Matrix& grad);

    /**
     * @brief Initialize the feed forward layer weights and biases
     * 
     * Uses Xavier/Glorot initialization for weights and small non-zero values for biases
     */
    void initialize_weights();
};