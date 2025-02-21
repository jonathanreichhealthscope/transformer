#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP

#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <functional>
#include <random>

#include "matrix.hpp"
#include "embeddings.hpp"  // Includes TokenEmbedding and PositionalEncoding
#include "attention.hpp"
#include "layer_norm.hpp"
#include "feed_forward.hpp"
#include "dropout.hpp"
#include "lm_head.hpp"
#include "config.hpp"
#include "cache.hpp"
#include "components.hpp"
#include "gradient_checkpoint.hpp"
#include "half_precision.hpp"
#include "memory_pool.hpp"
#include "phrase_types.hpp"
#include "tokenizer.hpp"

// Forward declarations
class TransformerLayer;
class LayerNorm;
class LanguageModelHead;
class Dropout;
class KVCache;
class Matrix;
class MultiHeadAttention;
class FeedForward;

// Add helper function declarations
void update_attention_parameters(MultiHeadAttention* attention, float learning_rate, const TransformerConfig& config);
void update_ffn_parameters(FeedForward* ffn, float learning_rate, const TransformerConfig& config);

// Add loss computation declarations
float compute_loss(const Matrix& output, const Matrix& target_distribution);
Matrix compute_loss_gradient(const Matrix& output, const Matrix& target_distribution);

// Make update_parameter_with_clip global functions instead of member functions
void update_parameter_with_clip(Matrix& param, const Matrix& grad, float learning_rate, const TransformerConfig& config);
void update_parameter_with_clip(Vector& param, const Vector& grad, float learning_rate, const TransformerConfig& config);

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
     * @brief Clears all cached states and resets layer components.
     */
    void clear_cache() {
        kv_cache.clear();
        if (self_attention) {
            self_attention->reset_state();
        }
        if (feed_forward) {
            feed_forward->reset_state();
        }
        if (attention_dropout) {
            attention_dropout->reset_mask();
        }
        if (ffn_dropout) {
            ffn_dropout->reset_mask();
        }
    }

    void set_training(bool mode) {
        training = mode;
    }

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
    LayerNorm* getLayerNorm() {
        return attention_ln.get();
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
    // Change from reference to value to avoid const issues
    TransformerConfig config;  // Store by value instead of reference

    // Components
    std::unique_ptr<TokenEmbedding> token_embedding;
    std::unique_ptr<PositionalEncoding> pos_encoding;
    std::vector<std::unique_ptr<TransformerLayer>> layers;
    std::unique_ptr<LayerNorm> final_ln;
    std::unique_ptr<LanguageModelHead> lm_head;
    std::unique_ptr<Dropout> dropout;

    // State
    bool training = true;
    Matrix hidden_states;
    Matrix last_hidden_states;
    std::vector<Matrix> m_layer_activations;
    std::vector<KVCache> m_kv_caches;
    std::vector<std::pair<size_t, size_t>> last_seq_boundaries;
    std::vector<int> last_input_tokens_;
    std::string last_input_query_;

    // Optimizer state
    std::vector<Matrix> momentum_buffers;
    std::vector<Matrix> velocity_buffers;
    size_t update_step = 0;
    std::optional<std::vector<Matrix>> parameter_grads;

    // Randomization helpers
    float get_dynamic_temperature(std::mt19937& gen) const {
        std::uniform_real_distribution<float> temp_dist(0.7f, 1.3f);
        return temp_dist(gen);
    }

    void add_random_noise(Matrix& logits, std::mt19937& gen) const {
        std::normal_distribution<float> noise_dist(0.0f, 0.1f);
        for (size_t i = 0; i < logits.cols(); i++) {
            logits(0, i) += noise_dist(gen);
        }
    }

    std::vector<float> apply_nucleus_sampling(
        const std::vector<float>& probabilities,
        float p,
        std::mt19937& gen
    ) const;

    void apply_random_boost(
        std::vector<float>& probabilities,
        std::mt19937& gen,
        float min_boost = 0.8f,
        float max_boost = 1.2f
    ) const;

    // Private methods
    void unscale_gradients(MultiHeadAttention::Gradients& grads, float scale);
    void unscale_gradients(FeedForward::Gradients& grads, float scale);
    Matrix compute_loss_gradients(const Matrix& logits, const std::vector<int>& targets);
    void backward_pass(const Matrix& output, const Matrix& target_distribution, float learning_rate);
    std::vector<Matrix>& parameter_gradients();
    void clear_gradients();

    // Helper methods for phrase prediction
    void boost_verb_probabilities(
        std::vector<float>& probabilities,
        const Tokenizer& tokenizer,
        std::mt19937* gen = nullptr
    );

    void boost_adjective_probabilities(
        std::vector<float>& probabilities,
        const Tokenizer& tokenizer,
        std::mt19937* gen = nullptr
    );

    bool is_likely_verb(const std::string& token) const;
    bool is_likely_adjective(const std::string& token) const;

    std::string extract_prediction(
        const Matrix& logits,
        PhraseType phrase_type,
        const Tokenizer& tokenizer,
        std::mt19937* gen = nullptr
    );

public:
    Transformer() = default;

    /**
     * @brief Initialize the weights of the transformer model
     */
    void initialize_weights();

    /**
     * @brief Sets the training mode for the transformer and all its components.
     * @param mode True for training mode, false for inference mode
     */
    void set_training(bool mode);

    /**
     * @brief Constructs a transformer model with the given configuration.
     * @param config Configuration parameters for the transformer
     */
    explicit Transformer(const TransformerConfig& config_);  // Just declaration, no implementation

    /**
     * @brief Performs the forward pass through the transformer.
     * @param input_tokens Input token sequence
     * @param original_query The original input query string
     * @param tokenizer The tokenizer instance to use for decoding
     * @param use_cache Whether to use key-value caching for inference
     * @return Output logits for each position
     */
    Matrix forward(const std::vector<int>& input_tokens, const std::string& original_query, const Tokenizer& tokenizer);

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

    bool verify_state() const {
        return token_embedding && pos_encoding && final_ln && lm_head && !layers.empty() &&
               std::all_of(layers.begin(), layers.end(), [](const auto& layer) { return layer != nullptr; });
    }

    bool is_training() const {
        return training;
    }

    void train_step(const std::vector<std::vector<int>>& input_tokens, 
                    const Matrix& target_distribution);

    void train_step(const std::vector<std::vector<int>>& input_tokens, 
                    const Matrix& target_distribution,
                    const Tokenizer& tokenizer);

    void save_checkpoint(const std::string& path);

    const std::vector<int>& get_last_input() const {
        return last_input_tokens_;
    }

    const std::string& get_last_query() const {
        return last_input_query_;
    }

    /**
     * @brief Updates model parameters using computed gradients.
     * @param learning_rate Learning rate for the update
     */
    void update_parameters(float learning_rate);

    /**
     * @brief Predicts the final phrase for a given input text without delimiters
     * @param input_text The input text without delimiters
     * @param tokenizer The tokenizer instance
     * @return A pair containing the predicted phrase and its type
     */
    std::pair<std::string, PhraseType> predict_final_phrase(
        const std::string& input_text,
        const Tokenizer& tokenizer
    );

    /**
     * @brief Predicts the most likely phrase type for the given input
     * @param input_text The input text
     * @param tokenizer The tokenizer instance
     * @return The predicted phrase type
     */
    PhraseType predict_phrase_type(
        const std::string& input_text,
        const Tokenizer& tokenizer
    );

private:
    /**
     * @brief Analyzes logits to determine the most likely phrase type
     * @param logits The output logits from the model
     * @param tokenizer The tokenizer instance
     * @return The predicted phrase type
     */
    PhraseType analyze_phrase_type(
        const Matrix& logits,
        const Tokenizer& tokenizer
    );
};

class PositionalEncoding;  // Forward declaration is enough since we include embeddings.hpp

#endif // TRANSFORMER_HPP