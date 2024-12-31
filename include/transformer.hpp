#pragma once
#include "attention.hpp"
#include "cache.hpp"
#include "components.hpp"
#include "embeddings.hpp"
#include "feed_forward.hpp"
#include "layernorm.hpp"
#include "lm_head.hpp"
#include "memory_pool.hpp"
#include "gradient_checkpoint.hpp"
#include "half_precision.hpp"
#include <functional>
#include <memory>
#include <vector>

// Forward declarations
class TokenEmbedding;
class PositionalEncoding;
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
  bool use_cuda;
  bool use_fp16;
  bool use_gradient_checkpointing;
  size_t memory_pool_size;

  TransformerConfig(size_t vocab_size = 50000, size_t max_seq_length = 2048,
                    size_t hidden_size = 768, size_t num_layers = 12,
                    size_t num_heads = 12);

  friend bool operator!=(const TransformerConfig &lhs,
                         const TransformerConfig &rhs) {
    return lhs.vocab_size != rhs.vocab_size ||
           lhs.max_seq_length != rhs.max_seq_length ||
           lhs.hidden_size != rhs.hidden_size ||
           lhs.num_layers != rhs.num_layers || lhs.num_heads != rhs.num_heads ||
           lhs.use_flash_attention != rhs.use_flash_attention ||
           lhs.use_rope != rhs.use_rope ||
           lhs.use_sliding_window != rhs.use_sliding_window ||
           lhs.window_size != rhs.window_size;
  }
};

class TransformerLayer {
private:
  std::unique_ptr<MultiHeadAttention> self_attention;
  std::unique_ptr<LayerNorm> attention_ln;
  std::unique_ptr<FeedForward> feed_forward;
  std::unique_ptr<LayerNorm> ffn_ln;
  KVCache kv_cache;
  TransformerConfig config;

public:
  virtual ~TransformerLayer() = default;
  TransformerLayer() = default;
  explicit TransformerLayer(const TransformerConfig &config);
  Matrix forward(const Matrix &x, const AttentionMask &mask = {});
  void clear_cache();
  void save(std::ostream &os) const {
    self_attention->save(os);
    attention_ln->save(os);
    feed_forward->save(os);
    ffn_ln->save(os);
  }
  static std::unique_ptr<TransformerLayer>
  create(const TransformerConfig &config) {
    return std::make_unique<TransformerLayer>(config);
  }
  void load(std::istream &is) {
    self_attention->load(is);
    attention_ln->load(is);
    feed_forward->load(is);
    ffn_ln->load(is);
  }
  Matrix backward(const Matrix &grad, const Matrix &input) const;
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

  TransformerLayer(const TransformerLayer& other) 
      : config(other.config), kv_cache(other.kv_cache) {
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
          config = other.config;
          kv_cache = other.kv_cache;
          
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

class Transformer {
private:
  std::vector<std::unique_ptr<TransformerLayer>> layers;
  std::unique_ptr<TokenEmbedding> token_embedding;
  std::unique_ptr<PositionalEncoding> pos_encoding;
  std::unique_ptr<LayerNorm> final_ln;
  std::unique_ptr<LanguageModelHead> lm_head;
  TransformerConfig config;
  bool cuda_initialized = false;

  Matrix compute_loss_gradients(const Matrix &logits,
                                const std::vector<int> &targets);
  void backward_pass(const std::vector<Matrix> &activations,
                     const Matrix &loss_grad);
  void update_parameters(float learning_rate);

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
  std::vector<Matrix> &parameters();
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
  Transformer(const Transformer& other);
  Transformer& operator=(const Transformer& other);
  
  // Move constructor and assignment operator
  Transformer(Transformer&& other) noexcept = default;
  Transformer& operator=(Transformer&& other) noexcept = default;
};