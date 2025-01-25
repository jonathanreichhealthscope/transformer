#pragma once
#include "attention.hpp"
#include "cache.hpp"
#include "components.hpp"
#include "config.hpp"
#include "dropout.hpp"
#include "embeddings.hpp"
#include "feed_forward.hpp"
#include "gradient_checkpoint.hpp"
#include "half_precision.hpp"
#include "layer_norm.hpp"
#include "lm_head.hpp"
#include "memory_pool.hpp"
#include <functional>
#include <memory>
#include <vector>

/**
 * @brief A single layer of the Transformer model implementing the standard Transformer architecture.
 * 
 * Each TransformerLayer consists of:
 * - Multi-head self-attention mechanism
 * - Layer normalization for attention
 * - Feed-forward neural network
 * - Layer normalization for feed-forward
 * - Dropout layers for regularization
 * - Key-Value cache for efficient inference
 */
class TransformerLayer {
  private:
    std::unique_ptr<MultiHeadAttention> self_attention;  ///< Multi-head self-attention mechanism
    std::unique_ptr<LayerNorm> attention_ln;            ///< Layer normalization for attention output
    std::unique_ptr<LayerNorm> ffn_ln;                 ///< Layer normalization for feed-forward output
    std::unique_ptr<FeedForward> feed_forward;         ///< Feed-forward neural network
    std::unique_ptr<Dropout> attention_dropout;        ///< Dropout for attention
    std::unique_ptr<Dropout> ffn_dropout;             ///< Dropout for feed-forward
    KVCache kv_cache;                                ///< Cache for key-value pairs in attention
    const TransformerConfig& config;                 ///< Reference to model configuration
    size_t layer_idx;                              ///< Index of this layer in the transformer
    bool training = false;                        ///< Whether the layer is in training mode

  public:
    virtual ~TransformerLayer() = default;
    TransformerLayer() = default;

    /**
     * @brief Constructs a transformer layer with the given configuration and layer index.
     * @param config_ Configuration parameters for the transformer
     * @param idx Index of this layer in the transformer stack
     */
    TransformerLayer(const TransformerConfig& config_, size_t idx);

    /**
     * @brief Performs the forward pass through the transformer layer.
     * @param input Input tensor of shape [batch_size, seq_len, hidden_size]
     * @param mask Attention mask to prevent attending to future tokens
     * @param kv_cache Optional key-value cache for efficient inference
     * @return Output tensor of shape [batch_size, seq_len, hidden_size]
     */
    Matrix forward(const Matrix& input, const AttentionMask& mask,
                   const std::optional<KVCache>& kv_cache = std::nullopt);

    /**
     * @brief Clears the key-value cache of this layer.
     */
    void clear_cache();

    /**
     * @brief Saves the layer's parameters to an output stream.
     * @param os Output stream to save to
     */
    void save(std::ostream& os) const {
        self_attention->save(os);
        attention_ln->save(os);
        feed_forward->save(os);
        ffn_ln->save(os);
    }

    /**
     * @brief Creates a new transformer layer with the given configuration.
     * @param config Configuration parameters for the transformer
     * @param idx Index of this layer in the transformer stack
     * @return Unique pointer to the created layer
     */
    static std::unique_ptr<TransformerLayer> create(const TransformerConfig& config, size_t idx) {
        return std::make_unique<TransformerLayer>(config, idx);
    }

    void load(std::istream& is) {
        self_attention = MultiHeadAttention::load(is, config);
    }
    Matrix backward(const Matrix& grad_output, const Matrix& input,
                    const Matrix& target_distribution = Matrix());
    Matrix backward_cuda(const Matrix& grad, const Matrix& input) const;
    std::vector<std::reference_wrapper<Matrix>> get_weights() {
        std::vector<std::reference_wrapper<Matrix>> weights;
        auto attention_weights = self_attention->get_weights();
        auto ff_weights = feed_forward->get_weights();

        weights.insert(weights.end(), attention_weights.begin(), attention_weights.end());
        weights.insert(weights.end(), ff_weights.begin(), ff_weights.end());

        return weights;
    }
    friend class Transformer;

    MultiHeadAttention* getAttention() {
        return self_attention.get();
    }
    FeedForward* getFeedForward() {
        return feed_forward.get();
    }
    void convert_to_fp16();

    TransformerLayer(const TransformerLayer& other)
        : config(other.config), kv_cache(other.kv_cache), layer_idx(other.layer_idx) {
        if (other.self_attention) {
            self_attention = std::make_unique<MultiHeadAttention>(*other.self_attention);
        }
        if (other.attention_ln) {
            attention_ln = std::make_unique<LayerNorm>(*other.attention_ln);
        }
        if (other.feed_forward) {
            feed_forward = std::make_unique<FeedForward>(*other.feed_forward);
        }
        if (other.ffn_ln) {
            ffn_ln = std::make_unique<LayerNorm>(*other.ffn_ln);
        }
    }

    TransformerLayer& operator=(const TransformerLayer& other) {
        if (this != &other) {
            kv_cache = other.kv_cache;
            layer_idx = other.layer_idx;

            if (other.self_attention) {
                self_attention = std::make_unique<MultiHeadAttention>(*other.self_attention);
            }
            if (other.attention_ln) {
                attention_ln = std::make_unique<LayerNorm>(*other.attention_ln);
            }
            if (other.feed_forward) {
                feed_forward = std::make_unique<FeedForward>(*other.feed_forward);
            }
            if (other.ffn_ln) {
                ffn_ln = std::make_unique<LayerNorm>(*other.ffn_ln);
            }
        }
        return *this;
    }

    void set_training(bool mode) {
        training = mode;
    }
};

/**
 * @brief Main Transformer model implementing the standard Transformer architecture.
 * 
 * The Transformer consists of:
 * - Token embedding layer
 * - Positional encoding
 * - Multiple transformer layers
 * - Final layer normalization
 * - Language model head for token prediction
 * 
 * Supports both training and inference modes, with features like:
 * - Key-Value caching for efficient inference
 * - CUDA acceleration
 * - Half-precision (FP16) computation
 * - Gradient checkpointing
 * - Various optimization algorithms
 */
class Transformer {
  private:
    TransformerConfig config;                          ///< Model configuration parameters
    std::unique_ptr<TokenEmbedding> token_embedding;   ///< Token embedding layer
    std::unique_ptr<PositionalEncoding> pos_encoding;  ///< Positional encoding layer
    std::vector<std::unique_ptr<TransformerLayer>> layers;  ///< Stack of transformer layers
    std::unique_ptr<LayerNorm> final_ln;              ///< Final layer normalization
    std::unique_ptr<LanguageModelHead> lm_head;       ///< Output layer for token prediction
    std::unique_ptr<Dropout> dropout;                 ///< Dropout layer
    bool cuda_initialized = false;                    ///< Whether CUDA has been initialized
    std::vector<int> last_input_tokens_;             ///< Store the last input tokens
    std::string last_input_query_;                   ///< Store the original input query

    // Cached states for backward pass
    Matrix hidden_states;
    Matrix last_hidden_states;
    std::vector<Matrix> m_layer_activations;

    // KV cache for inference
    std::vector<KVCache> m_kv_caches;

    // Optimizer state
    std::vector<Matrix> momentum_buffers;
    std::vector<Matrix> velocity_buffers;
    size_t update_step = 0;

    // Parameter gradients
    std::optional<std::vector<Matrix>> parameter_grads;

    // Private methods
    Matrix compute_loss_gradients(const Matrix& logits, const std::vector<int>& targets);
    void backward_pass(const std::vector<Matrix>& activations, const Matrix& loss_grad);
    void update_parameters(float learning_rate);

    // Get parameter gradients
    std::vector<Matrix>& parameter_gradients() {
        if (!parameter_grads.has_value()) {
            parameter_grads = std::vector<Matrix>();
            // Initialize gradients for all parameters
            auto& params = parameters();
            parameter_grads->reserve(params.size());
            for (const auto& param : params) {
                parameter_grads->emplace_back(param.rows(), param.cols(), 0.0f);
            }
        }
        return parameter_grads.value();
    }

    bool training = true;

  public:
    Transformer() = default;

    /**
     * @brief Constructs a transformer model with the given configuration.
     * @param config Configuration parameters for the transformer
     */
    explicit Transformer(const TransformerConfig& config);

    /**
     * @brief Performs the forward pass through the transformer.
     * @param input_tokens Input token sequence
     * @param use_cache Whether to use key-value caching for inference
     * @return Output logits for each position
     */
    Matrix forward(const std::vector<int>& input_tokens, bool use_cache = false);

    /**
     * @brief Trains the transformer on the given dataset.
     * @param input_tokens Batch of input token sequences
     * @param target_tokens Batch of target token sequences
     * @param num_epochs Number of training epochs
     * @param learning_rate Learning rate for optimization
     */
    void train(const std::vector<std::vector<int>>& input_tokens,
               const std::vector<std::vector<int>>& target_tokens, size_t num_epochs,
               float learning_rate);

    /**
     * @brief Saves the model parameters to a file.
     * @param path Path to save the model to
     */
    void save_model(const std::string& path) const;

    /**
     * @brief Loads a model from a file.
     * @param path Path to load the model from
     * @return The loaded transformer model
     */
    static Transformer load_model(const std::string& path);

    /**
     * @brief Clears all key-value caches in the model.
     */
    void clear_kv_cache();

    Matrix backward(const Matrix& grad, const Matrix& activation, size_t layer_idx);
    Matrix backward_cuda(const Matrix& grad, const Matrix& activation, size_t layer_idx);
    std::vector<Matrix>& parameters();
    void save(std::ostream& os) const;
    void load(std::istream& is);

    std::vector<std::vector<std::reference_wrapper<Matrix>>> get_layer_weights() const {
        std::vector<std::vector<std::reference_wrapper<Matrix>>> all_weights;
        for (const auto& layer : layers) {
            all_weights.push_back(layer->get_weights());
        }
        return all_weights;
    }

    friend class QuantizationAwareTraining;

    const TransformerConfig& getConfig() const {
        return config;
    }
    const std::vector<std::unique_ptr<TransformerLayer>>& getLayers() const {
        return layers;
    }
    std::vector<std::unique_ptr<TransformerLayer>>& getLayers() {
        return layers;
    }
    virtual ~Transformer();

    // Add copy constructor and assignment operator
    Transformer(const Transformer& other);
    Transformer& operator=(const Transformer& other);

    // Move constructor and assignment operator
    Transformer(Transformer&& other) noexcept = default;
    Transformer& operator=(Transformer&& other) noexcept = default;

    // Keep the original backward method for single sample training
    void backward(const Matrix& grad_output, const std::vector<int>& input_tokens, float learning_rate);
    
    // Add new backward method for batch training
    void backward(std::vector<Matrix>& outputs, const Matrix& target_distribution, float learning_rate);

    const Matrix& get_hidden_states() const {
        return hidden_states;
    }
    LanguageModelHead* get_lm_head() {
        return lm_head.get();
    }
    void set_lm_head(std::unique_ptr<LanguageModelHead> head) {
        lm_head = std::move(head);
    }

    void set_training(bool mode) {
        training = mode;
        for (auto& layer : layers) {
            layer->set_training(mode);
        }
    }

    bool verify_state() const {
        return token_embedding && pos_encoding && final_ln && lm_head && !layers.empty() &&
               std::all_of(layers.begin(), layers.end(),
                           [](const auto& layer) { return layer != nullptr; });
    }

    bool is_training() const {
        return training;
    }

    void train_step(const std::vector<std::vector<int>>& input_tokens, 
                    const Matrix& target_distribution);

    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);

    const std::vector<int>& get_last_input() const {
        return last_input_tokens_;
    }

    const std::string& get_last_query() const {
        return last_input_query_;
    }
};