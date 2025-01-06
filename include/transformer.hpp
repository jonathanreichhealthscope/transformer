#pragma once
#include "attention.hpp"
#include "cache.hpp"
#include "components.hpp"
#include "embeddings.hpp"
#include "feed_forward.hpp"
#include "gradient_checkpoint.hpp"
#include "half_precision.hpp"
#include "layer_norm.hpp"
#include "lm_head.hpp"
#include "memory_pool.hpp"
#include "dropout.hpp"
#include <functional>
#include <memory>
#include <vector>

// Forward declarations
class TransformerLayer;

class TransformerConfig {
public:
  size_t vocab_size;
  size_t max_seq_length;
  size_t hidden_size;
  size_t num_layers;
  size_t num_heads;
  size_t head_dim;
  size_t intermediate_size;
  float dropout_prob;
  bool use_flash_attention;
  bool use_rope;
  bool use_sliding_window;
  size_t window_size;
  bool use_gqa;
  size_t num_kv_heads;
  bool use_fp16;
  bool use_gradient_checkpointing;
  size_t memory_pool_size;
  size_t batch_size;
  size_t num_epochs;
  float dropout_rate;
  float weight_decay;
  struct {
    std::string save_directory;
    std::string model_name;
    size_t checkpoint_frequency;
  } paths;
  bool load_from_checkpoint;
  std::string checkpoint_to_load;

  TransformerConfig(size_t vocab_size = 32000, size_t max_seq_length = 512,
                   size_t hidden_size = 768, size_t num_layers = 12,
                   size_t num_heads = 12, size_t batch_size = 32,
                   size_t num_epochs = 10);

  friend bool operator!=(const TransformerConfig &lhs,
                         const TransformerConfig &rhs) {
    return lhs.vocab_size != rhs.vocab_size ||
           lhs.max_seq_length != rhs.max_seq_length ||
           lhs.hidden_size != rhs.hidden_size ||
           lhs.num_layers != rhs.num_layers || lhs.num_heads != rhs.num_heads ||
           lhs.use_flash_attention != rhs.use_flash_attention ||
           lhs.use_rope != rhs.use_rope ||
           lhs.use_sliding_window != rhs.use_sliding_window ||
           lhs.window_size != rhs.window_size ||
           lhs.batch_size != rhs.batch_size ||
           lhs.num_epochs != rhs.num_epochs;
  }
};

class TransformerLayer {
private:
  std::unique_ptr<MultiHeadAttention> self_attention;
  std::unique_ptr<LayerNorm> attention_ln;
  std::unique_ptr<LayerNorm> ffn_ln;
  std::unique_ptr<FeedForward> feed_forward;
  std::unique_ptr<Dropout> attention_dropout;
  std::unique_ptr<Dropout> ffn_dropout;
  KVCache kv_cache;
  const TransformerConfig& config;
  size_t layer_idx;
  bool training = false;

public:
  virtual ~TransformerLayer() = default;
  TransformerLayer() = default;
  TransformerLayer(const TransformerConfig& config_, size_t idx);
  Matrix forward(const Matrix &input, const AttentionMask &mask,
                 const std::optional<KVCache> &kv_cache = std::nullopt);
  void clear_cache();
  void save(std::ostream &os) const {
    self_attention->save(os);
    attention_ln->save(os);
    feed_forward->save(os);
    ffn_ln->save(os);
  }
  static std::unique_ptr<TransformerLayer>
  create(const TransformerConfig &config, size_t idx) {
    return std::make_unique<TransformerLayer>(config, idx);
  }
  void load(std::istream& is) {
    self_attention = MultiHeadAttention::load(is, config);
  }
  Matrix backward(const Matrix &grad_output, const Matrix &input,
                 const Matrix &target_distribution = Matrix());
  Matrix backward_cuda(const Matrix &grad, const Matrix &input) const;
  std::vector<std::reference_wrapper<Matrix>> get_weights() {
    std::vector<std::reference_wrapper<Matrix>> weights;
    auto attention_weights = self_attention->get_weights();
    auto ff_weights = feed_forward->get_weights();

    weights.insert(weights.end(), attention_weights.begin(),
                   attention_weights.end());
    weights.insert(weights.end(), ff_weights.begin(), ff_weights.end());

    return weights;
  }
  friend class Transformer;

  MultiHeadAttention *getAttention() { return self_attention.get(); }
  FeedForward *getFeedForward() { return feed_forward.get(); }
  void convert_to_fp16();

  TransformerLayer(const TransformerLayer &other)
      : config(other.config), 
        kv_cache(other.kv_cache),
        layer_idx(other.layer_idx) {
    if (other.self_attention) {
      self_attention =
          std::make_unique<MultiHeadAttention>(*other.self_attention);
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

  TransformerLayer &operator=(const TransformerLayer &other) {
    if (this != &other) {
      kv_cache = other.kv_cache;
      layer_idx = other.layer_idx;

      if (other.self_attention) {
        self_attention =
            std::make_unique<MultiHeadAttention>(*other.self_attention);
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

class Transformer {
private:
  TransformerConfig config;
  std::unique_ptr<TokenEmbedding> token_embedding;
  std::unique_ptr<PositionalEncoding> pos_encoding;
  std::vector<std::unique_ptr<TransformerLayer>> layers;
  std::unique_ptr<LayerNorm> final_ln;
  std::unique_ptr<LanguageModelHead> lm_head;
  bool cuda_initialized = false;
  
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
  Matrix compute_loss_gradients(const Matrix &logits, const std::vector<int> &targets);
  void backward_pass(const std::vector<Matrix> &activations, const Matrix &loss_grad);
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

  bool training = false;

public:
  Transformer() = default;
  explicit Transformer(const TransformerConfig &config);
  Matrix forward(const std::vector<int> &input_tokens, bool use_cache = false);
  Matrix forward_cuda(const std::vector<int> &input_tokens,
                      bool use_cache = false);
  void train(const std::vector<std::vector<int>> &input_tokens,
             const std::vector<std::vector<int>> &target_tokens,
             size_t num_epochs, float learning_rate);
  void save_model(const std::string &path) const;
  static Transformer load_model(const std::string &path);
  void clear_kv_cache();
  Matrix backward(const Matrix &grad, const Matrix &activation,
                  size_t layer_idx);
  Matrix backward_cuda(const Matrix &grad, const Matrix &activation,
                       size_t layer_idx);
  std::vector<Matrix>& parameters();
  void save(std::ostream &os) const;
  void load(std::istream &is);

  std::vector<std::vector<std::reference_wrapper<Matrix>>>
  get_layer_weights() const {
    std::vector<std::vector<std::reference_wrapper<Matrix>>> all_weights;
    for (const auto &layer : layers) {
      all_weights.push_back(layer->get_weights());
    }
    return all_weights;
  }

  friend class TransformerTrainer;
  friend class QuantizationAwareTraining;

  const TransformerConfig &getConfig() const { return config; }
  const std::vector<std::unique_ptr<TransformerLayer>> &getLayers() const {
    return layers;
  }
  std::vector<std::unique_ptr<TransformerLayer>> &getLayers() { return layers; }
  virtual ~Transformer();

  // Add copy constructor and assignment operator
  Transformer(const Transformer &other);
  Transformer &operator=(const Transformer &other);

  // Move constructor and assignment operator
  Transformer(Transformer &&other) noexcept = default;
  Transformer &operator=(Transformer &&other) noexcept = default;

  void backward(const Matrix &grad_output, const std::vector<int> &input_tokens, float learning_rate);

  const Matrix& get_hidden_states() const { return hidden_states; }
  LanguageModelHead* get_lm_head() { return lm_head.get(); }
  void set_lm_head(std::unique_ptr<LanguageModelHead> head) { lm_head = std::move(head); }

  void set_training(bool mode) { 
    training = mode;
    for (auto& layer : layers) {
      layer->set_training(mode);
    }
  }

  bool verify_state() const {
    return token_embedding && pos_encoding && final_ln && lm_head && 
           !layers.empty() && std::all_of(layers.begin(), layers.end(), 
           [](const auto& layer) { return layer != nullptr; });
  }
};