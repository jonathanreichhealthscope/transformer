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
    // Parameter structure to hold all weights and biases
    struct Parameters {
        Matrix ff1_weights;
        Matrix ff2_weights;
        FloatVector ff1_bias;
        FloatVector ff2_bias;
    };

    // Gradient structure to hold all gradients
    struct Gradients {
        Matrix ff1_grad;         // Gradient for first layer weights
        Matrix ff2_grad;         // Gradient for second layer weights
        FloatVector ff1_bias_grad;  // Gradient for first layer bias
        FloatVector ff2_bias_grad;  // Gradient for second layer bias
    };

    Parameters params_;
    Gradients grads_;

    // Cache for intermediate values
    Matrix intermediate_cache;    ///< Cache for intermediate activations during forward pass
    Matrix input_cache_;         ///< Cache for input during backward pass
    Matrix dropout_mask_;        ///< Dropout mask for training
    
    float dropout_prob;           ///< Dropout probability during training
    bool training_ = true;        ///< Training mode control

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

    // Update accessor methods to use Parameters structure
    Matrix& get_ff1_weights() { return params_.ff1_weights; }
    Matrix& get_ff2_weights() { return params_.ff2_weights; }
    FloatVector& getBias1() { return params_.ff1_bias; }
    FloatVector& getBias2() { return params_.ff2_bias; }

    // Add const versions
    const Matrix& get_ff1_weights() const { return params_.ff1_weights; }
    const Matrix& get_ff2_weights() const { return params_.ff2_weights; }
    const FloatVector& getBias1() const { return params_.ff1_bias; }
    const FloatVector& getBias2() const { return params_.ff2_bias; }

    // Add parameter accessors
    Parameters& parameters() { return params_; }
    Gradients& param_gradients() { return grads_; }
    const Parameters& parameters() const { return params_; }
    const Gradients& param_gradients() const { return grads_; }

    // Update get_weights to use Parameters structure
    std::vector<std::reference_wrapper<Matrix>> get_weights() {
        return {std::ref(params_.ff1_weights), std::ref(params_.ff2_weights)};
    }

    // Update copy constructor to use Parameters/Gradients
    FeedForward(const FeedForward& other)
        : params_(other.params_), grads_(other.grads_),
          intermediate_cache(other.intermediate_cache),
          input_cache_(other.input_cache_),
          dropout_mask_(other.dropout_mask_),
          dropout_prob(other.dropout_prob),
          training_(other.training_) {}

    // Update assignment operator to use Parameters/Gradients
    FeedForward& operator=(const FeedForward& other) {
        if (this != &other) {
            params_ = other.params_;
            grads_ = other.grads_;
            intermediate_cache = other.intermediate_cache;
            input_cache_ = other.input_cache_;
            dropout_mask_ = other.dropout_mask_;
            dropout_prob = other.dropout_prob;
            training_ = other.training_;
        }
        return *this;
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

    friend class Transformer;
    friend class TransformerLayer;
};