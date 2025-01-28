#pragma once
#include "components.hpp"
#include "cuda_utils.hpp"
#include "layer_norm.hpp"
#include "tiktoken_tokenizer.hpp"
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <deque>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#endif

/**
 * @brief Language model head for token prediction in transformer models.
 * 
 * The LanguageModelHead class transforms hidden states into logits over the vocabulary,
 * enabling token prediction for language modeling tasks. Features include:
 * - Linear projection to vocabulary size
 * - Bias terms for each token
 * - Adaptive token frequency tracking
 * - Adam optimizer integration
 * - Dropout regularization
 */
class LanguageModelHead {
  private:
    Matrix projection;                    ///< Projection matrix to vocabulary space
    Vector bias;                         ///< Bias terms for each token
    float dropout_prob;                  ///< Dropout probability during training
    size_t vocab_size_;                  ///< Size of the vocabulary
    size_t hidden_size_;                 ///< Size of input hidden states
    Matrix hidden_states;                ///< Cached hidden states for backward pass
    Matrix hidden_states_;               ///< Cached hidden states for forward pass
    std::vector<float> token_frequencies; ///< Tracked frequencies of token usage
    float pruning_threshold;
    std::vector<unsigned char> active_tokens;  // Changed from vector<bool> to vector<unsigned char>
    std::vector<int> active_token_indices;     // List of indices of active tokens
    size_t training_steps;
    bool is_training_;  // Add training state member variable
    
    // Adam optimizer state
    Matrix m_proj;  // Momentum for projection
    Matrix v_proj;  // RMSprop for projection
    Vector m_bias;  // Momentum for bias
    Vector v_bias;  // RMSprop for bias
    size_t t;      // Time step
    float beta1;   // Momentum parameter
    float beta2;   // RMSprop parameter
    float eps;     // Small constant for numerical stability
    
    // Learning rate adaptation
    float current_lr;  // Current learning rate
    float min_lr;      // Minimum learning rate
    float max_lr;      // Maximum learning rate
    float lr_decay;    // Learning rate decay factor
    float lr_growth;   // Learning rate growth factor
    std::deque<float> loss_history;
    static constexpr size_t LOSS_HISTORY_SIZE = 100;
    float prev_loss = std::numeric_limits<float>::infinity();
    
    // Add the update_learning_rate function declaration
    void update_learning_rate(float current_loss);
    
    // Pinned memory for efficient GPU transfers
    float* h_projection = nullptr;
    float* h_bias = nullptr;

    // Device memory buffers
    float* d_projection = nullptr;  // Device copy of projection matrix
    float* d_bias = nullptr;       // Device copy of bias
    half* d_projection_fp16 = nullptr;  // FP16 version of projection
    half* d_hidden_states_fp16 = nullptr;  // FP16 version of input
    half* d_output_fp16 = nullptr;  // FP16 intermediate output
    float* d_output = nullptr;      // Final FP32 output

    // Add new member variables
    std::unique_ptr<LayerNorm> layer_norm;  ///< Layer normalization
    std::shared_ptr<TiktokenTokenizer> tokenizer;  ///< Tokenizer instance

    /**
     * @brief Computes gradients for the linear projection.
     * @param grad_output Gradient of the loss with respect to the output
     */
    void backward_linear(const Matrix& grad_output);

    /**
     * @brief Implementation of the forward pass computation.
     * @param hidden_states Input hidden states
     * @return Output logits over vocabulary
     */
    Matrix forward_impl(const Matrix& hidden_states);

    void update_active_tokens();

#ifdef USE_CUDA
    // CUDA streams and synchronization
    cudaStream_t compute_stream;

    // Device memory for active tokens and indices
    unsigned char* d_active_tokens = nullptr;
    int* d_active_token_indices = nullptr;

    // Maximum batch size for memory allocation
    static constexpr size_t max_batch_size = 4096;  // Adjust based on your needs

    // CUDA kernel launchers
    __host__ void launch_convert_to_fp16(half* output, const float* input, size_t size);
    __host__ void launch_convert_and_expand_vocab(
        float* output, const half* input, size_t batch_size, size_t vocab_size, size_t active_vocab_size);

    cublasHandle_t cublas_handle;
#endif

    // Helper methods
    void bias_completion_format(Matrix& logits);

  public:
    /**
     * @brief Constructs a language model head.
     * @param hidden_size Size of input hidden states
     * @param vocab_size Size of the vocabulary
     */
    LanguageModelHead(size_t hidden_size, size_t vocab_size);

    ~LanguageModelHead();  // Just declare it here

    /**
     * @brief Performs the forward pass, computing logits from hidden states.
     * @param hidden_states Input hidden states
     * @return Matrix of logits over vocabulary
     */
    Matrix forward(const Matrix& hidden_states, bool training = false);

    /**
     * @brief Performs the backward pass with Adam optimization.
     * @param grad_output Gradient of the loss with respect to the output
     * @param hidden_states Original input hidden states
     * @return Gradient with respect to the input
     */
    Matrix backward_pass(const Matrix& grad_output, const Matrix& hidden_states);

    /**
     * @brief Saves the model head to a stream.
     * @param os Output stream to save to
     */
    void save(std::ostream& os) const {
        projection.save(os);
        bias.save(os);
        os.write(reinterpret_cast<const char*>(&dropout_prob), sizeof(dropout_prob));
    }

    /**
     * @brief Loads a model head from a stream.
     * @param is Input stream to load from
     * @return Unique pointer to loaded model head
     */
    static std::unique_ptr<LanguageModelHead> load(std::istream& is) {
        auto lm_head = std::make_unique<LanguageModelHead>(0, 0); // Temporary sizes
        lm_head->projection = Matrix::load(is);
        lm_head->bias = Vector::load(is);
        is.read(reinterpret_cast<char*>(&lm_head->dropout_prob), sizeof(lm_head->dropout_prob));
        return lm_head;
    }

    /**
     * @brief Gets references to trainable parameters.
     * @return Vector of parameter references
     */
    std::vector<std::reference_wrapper<Matrix>> get_parameters() {
        std::vector<std::reference_wrapper<Matrix>> params;
        params.push_back(std::ref(projection));
        // Note: We'll need to handle bias separately since it's a Vector
        return params;
    }

    /**
     * @brief Gets the bias vector.
     * @return Reference to bias vector
     */
    Vector& get_bias() {
        return bias;
    }

    /**
     * @brief Projects hidden states to vocabulary space.
     * @param hidden_states Input hidden states
     * @return Matrix of logits over vocabulary
     */
    Matrix project_to_vocab(const Matrix& hidden_states);

    /**
     * @brief Performs backward pass with optional target distribution.
     * @param grad_output Gradient of the loss with respect to the output
     * @param target_distribution Optional target distribution for distillation
     * @return Gradient with respect to the input
     */
    Matrix backward(const Matrix& grad_output, const Matrix& target_distribution = Matrix());

    /**
     * @brief Updates token frequencies based on observed tokens.
     * @param tokens Vector of token indices observed in the current batch
     */
    void update_token_frequencies(const std::vector<int>& tokens);

    /**
     * @brief Prunes vocabulary by removing infrequently used tokens.
     * @param min_frequency_threshold Minimum frequency threshold for keeping tokens
     */
    void prune_vocabulary(float min_frequency_threshold = 1e-5);

    void set_training(bool training_mode);  // Add setter for training mode

    // Add tokenizer setter
    void set_tokenizer(std::shared_ptr<TiktokenTokenizer> tok) { tokenizer = tok; }
};