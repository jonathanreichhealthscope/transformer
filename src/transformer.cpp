#include "../include/transformer.hpp"'
#include "../include/cuda/cublas_check.cuh"
#include "../include/cuda/cuda_check.cuh"
#include "../include/logger.hpp"
#include <cublas_v2.h>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <stdexcept>

extern cublasHandle_t cublas_handle;

// TransformerConfig implementation
TransformerConfig::TransformerConfig(size_t vocab_size, size_t max_seq_length,
                                     size_t hidden_size, size_t num_layers,
                                     size_t num_heads)
    : vocab_size(vocab_size), max_seq_length(max_seq_length),
      hidden_size(hidden_size), num_layers(num_layers), num_heads(num_heads),
      head_dim(hidden_size / num_heads), intermediate_size(4 * hidden_size),
      dropout_prob(0.1f), use_flash_attention(true), use_rope(true),
      use_sliding_window(false), window_size(512), use_gqa(false),
      num_kv_heads(num_heads), use_cuda(true) {
  if (hidden_size % num_heads != 0) {
    throw std::invalid_argument(
        "Hidden size must be divisible by number of heads");
  }
}

// TransformerLayer implementation
TransformerLayer::TransformerLayer(const TransformerConfig &config)
    : kv_cache(config.max_seq_length), config(config) {
  // Initialize attention layer
  self_attention = std::make_unique<MultiHeadAttention>(
      config.hidden_size, config.num_heads, config.head_dim,
      config.dropout_prob, config.use_flash_attention, config.use_rope,
      config.use_sliding_window, config.window_size, config.use_gqa,
      config.num_kv_heads);

  // Initialize layer normalization
  attention_ln = std::make_unique<LayerNorm>(config.hidden_size);
  ffn_ln = std::make_unique<LayerNorm>(config.hidden_size);

  // Initialize feed-forward network
  feed_forward = std::make_unique<FeedForward>(
      config.hidden_size, config.intermediate_size, config.dropout_prob);
}

Matrix TransformerLayer::forward(const Matrix &x, const AttentionMask &mask) {
  // Pre-layer normalization
  Matrix normalized = attention_ln->forward(x);
  // Self-attention with residual connection
  Matrix attention_output = self_attention->forward(
      normalized, mask, std::make_optional(std::ref(kv_cache)));
  Matrix residual = x + attention_output;
  std::cout << "Residual: " << *residual.data() << std::endl;
  // Feed-forward with residual connection
  normalized = ffn_ln->forward(residual);
  Matrix ffn_output = feed_forward->forward(normalized);

  return residual + ffn_output;
}

void TransformerLayer::clear_cache() { kv_cache.clear(); }

void TransformerLayer::convert_to_fp16() {
#ifdef USE_CUDA
  if (self_attention) {
    auto weights = self_attention->get_weights();
    for (auto &weight : weights) {
      HalfPrecisionTraining::convert_to_fp16(weight);
    }
  }
  if (feed_forward) {
    auto weights = feed_forward->get_weights();
    for (auto &weight : weights) {
      HalfPrecisionTraining::convert_to_fp16(weight);
    }
  }
#endif
}

// Transformer implementation
Transformer::Transformer(const TransformerConfig &config) : config(config) {
  if (config.use_cuda) {
    initialize_cuda();
    cuda_initialized = true;
  }

  // Initialize token embedding with memory pooling
  token_embedding =
      std::make_unique<TokenEmbedding>(config.vocab_size, config.hidden_size);

  // Initialize positional encoding
  pos_encoding = std::make_unique<PositionalEncoding>(config.max_seq_length,
                                                      config.hidden_size);

  // Initialize transformer layers
  layers.reserve(config.num_layers);
  for (size_t i = 0; i < config.num_layers; ++i) {
    layers.push_back(std::make_unique<TransformerLayer>(config));
  }

  // Initialize final layer normalization
  final_ln = std::make_unique<LayerNorm>(config.hidden_size);

  // Enable half-precision training if configured
  if (config.use_fp16) {
    for (auto &layer : layers) {
      layer->convert_to_fp16();
    }
  }
}

Matrix Transformer::forward(const std::vector<int> &input_tokens,
                            bool use_cache) {
  if (config.use_cuda) {
    std::cout << "Using CUDA for forward pass" << std::endl;
    return forward_cuda(input_tokens);
  }
  // Use memory pool for embeddings
  size_t embed_size = input_tokens.size() * config.hidden_size;
  float *embed_data = MemoryPool::allocate_static(embed_size * sizeof(float));
  Matrix embeddings(input_tokens.size(), config.hidden_size, embed_data);

  // Get embeddings using cuBLAS for matrix operations
  token_embedding->forward_cuda(input_tokens, embeddings);

  // Add positional encodings
  Matrix position_ids(input_tokens.size(), 1);
  for (size_t i = 0; i < input_tokens.size(); ++i) {
    position_ids(i, 0) = static_cast<float>(i);
  }
  embeddings += pos_encoding->forward(position_ids);

  // Create attention mask if needed
  AttentionMask mask;
  if (!use_cache) {
    mask = AttentionMask::create_causal_mask(input_tokens.size());
  }

  // Forward pass through layers with gradient checkpointing
  hidden_states = embeddings;
  for (size_t i = 0; i < layers.size(); ++i) {
    // Save activation for gradient checkpointing
    GradientCheckpoint::save_activation(hidden_states, i);

    hidden_states = layers[i]->forward(hidden_states, mask);

    // Convert to FP16 if enabled
    if (config.use_fp16) {
      HalfPrecisionTraining::convert_to_fp16(hidden_states);
    }
  }

  // Final layer normalization
  hidden_states = final_ln->forward(hidden_states);

  // Store the hidden states for backward pass
  last_hidden_states = hidden_states;

  // Free memory pool allocation
  MemoryPool::deallocate_static(embed_data, embed_size * sizeof(float));

  return hidden_states;
}

void Transformer::train(const std::vector<std::vector<int>> &input_tokens,
                        const std::vector<std::vector<int>> &target_tokens,
                        size_t num_epochs, float learning_rate) {
  const size_t batch_size = 32; // Fixed batch size

  for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
// Process batches
#pragma omp parallel for
    for (size_t i = 0; i < input_tokens.size(); i += batch_size) {
      size_t batch_end = std::min(i + batch_size, input_tokens.size());

      // Create batch
      std::vector<std::vector<int>> input_batch(
          input_tokens.begin() + i, input_tokens.begin() + batch_end);
      std::vector<std::vector<int>> target_batch(
          target_tokens.begin() + i, target_tokens.begin() + batch_end);

      // Forward pass
      std::vector<Matrix> activations;
      Matrix logits = forward(input_batch[0], &activations);

      // Compute loss and gradients
      Matrix loss_grad = compute_loss_gradients(logits, target_batch[0]);

      // Backward pass
      backward_pass(activations, loss_grad);

      // Update parameters using optimizer (you'll need to implement this)
      update_parameters(learning_rate);
    }
  }
}

Matrix Transformer::compute_loss_gradients(const Matrix &logits,
                                           const std::vector<int> &targets) {
  const size_t batch_size = logits.rows();
  const size_t vocab_size = logits.cols();
  Matrix gradients(batch_size, vocab_size);

  // For each sequence position
  for (size_t i = 0; i < batch_size; ++i) {
    // Compute softmax probabilities
    std::vector<float> probs(vocab_size);
    float max_logit = -std::numeric_limits<float>::infinity();

    // Find max logit for numerical stability
    for (size_t j = 0; j < vocab_size; ++j) {
      max_logit = std::max(max_logit, logits(i, j));
    }

    // Compute softmax denominator
    float sum_exp = 0.0f;
    for (size_t j = 0; j < vocab_size; ++j) {
      probs[j] = std::exp(logits(i, j) - max_logit);
      sum_exp += probs[j];
    }

    // Normalize and compute gradients
    for (size_t j = 0; j < vocab_size; ++j) {
      probs[j] /= sum_exp;
      // Gradient is (probability - 1) for correct class, probability for others
      gradients(i, j) = probs[j];
    }
    gradients(i, targets[i]) -= 1.0f; // Subtract 1 from target class
  }

  return gradients;
}

void Transformer::backward_pass(const std::vector<Matrix> &activations,
                                const Matrix &loss_grad) {
  Matrix current_grad = loss_grad;

  // Convert gradients to FP16 if enabled
  if (config.use_fp16) {
    HalfPrecisionTraining::convert_to_fp16(current_grad);
  }

  // Backward through final layer norm using cuBLAS
  current_grad = final_ln->backward_cuda(current_grad, activations.back());

  // Backward through layers in reverse order
  for (int i = layers.size() - 1; i >= 0; --i) {
    // Retrieve checkpointed activation
    Matrix activation = GradientCheckpoint::get_activation(i);

    if (config.use_cuda) {
      current_grad = layers[i]->backward_cuda(current_grad, activation);
    } else {
      current_grad = layers[i]->backward(current_grad, activation);
    }

    // Convert gradients back to FP32 if needed
    if (config.use_fp16) {
      HalfPrecisionTraining::convert_to_fp32(current_grad);
    }
  }
}

void Transformer::update_parameters(float learning_rate) {
  // Get all trainable parameters and their gradients
  auto &params = this->parameters();

  // Simple SGD update
  for (size_t i = 0; i < params.size(); ++i) {
    Matrix &param = params[i];
    // Assuming you have stored gradients somewhere
    // Matrix& grad = parameter_gradients[i];

    // Update rule: param = param - learning_rate * grad
    for (size_t row = 0; row < param.rows(); ++row) {
      for (size_t col = 0; col < param.cols(); ++col) {
        // param(row, col) -= learning_rate * grad(row, col);
        // For now, just add a placeholder update
        param(row, col) -=
            learning_rate * 0.01f; // Replace with actual gradient
      }
    }
  }
}

void Transformer::save_model(const std::string &path) const {
  std::ofstream os(path, std::ios::binary);
  if (!os) {
    throw std::runtime_error("Failed to open file for saving model");
  }

  // Save config
  os.write(reinterpret_cast<const char *>(&config), sizeof(config));

  // Save embeddings
  token_embedding->save(os);
  pos_encoding->save(os);

  // Save layers
  for (const auto &layer : layers) {
    layer->save(os);
  }

  // Save final layer norm
  final_ln->save(os);
}

Transformer Transformer::load_model(const std::string &path) {
  std::ifstream is(path, std::ios::binary);
  if (!is) {
    throw std::runtime_error("Failed to open file for loading model");
  }

  // Load config
  TransformerConfig config;
  is.read(reinterpret_cast<char *>(&config), sizeof(config));

  // Create transformer with loaded config
  Transformer transformer(config);

  // Load embeddings
  transformer.token_embedding = TokenEmbedding::load(is);
  transformer.pos_encoding = PositionalEncoding::load(is);

  // Load layers
  transformer.layers.clear();
  for (size_t i = 0; i < config.num_layers; ++i) {
    auto layer = TransformerLayer::create(config);
    layer->load(is);
    transformer.layers.push_back(std::move(layer));
  }

  // Load final layer norm
  transformer.final_ln = LayerNorm::load(is);

  return transformer;
}

void Transformer::clear_kv_cache() {
  for (auto &layer : layers) {
    layer->clear_cache();
  }
}

Matrix Transformer::backward(const Matrix &grad, const Matrix &activation,
                             size_t layer_idx) {
  if (layer_idx >= layers.size()) {
    throw std::out_of_range("Layer index out of range");
  }

  // Compute gradients for the current layer
  Matrix layer_grad = grad;

  // Backward through layer normalization
  if (layer_idx == layers.size() - 1) {
    layer_grad = final_ln->backward(layer_grad, activation);
  }

  // Backward through transformer layer
  // Note: This would require implementing backward methods in TransformerLayer
  // and its components (attention, feed-forward, etc.)

  return layer_grad;
}

Matrix Transformer::backward_cuda(const Matrix &grad, const Matrix &activation,
                                  size_t layer_idx) {
#ifdef USE_CUDA
  if (layer_idx >= layers.size()) {
    throw std::out_of_range("Layer index out of range");
  }

  Matrix current_grad = grad;

  // Convert gradients to FP16 if enabled
  if (config.use_fp16) {
    HalfPrecisionTraining::convert_to_fp16(current_grad);
  }

  // Backward through final layer norm using CUDA
  if (layer_idx == layers.size() - 1) {
    current_grad = final_ln->backward_cuda(current_grad, activation);
  }

  // Backward through layer using CUDA
  Matrix layer_grad =
      layers[layer_idx]->backward_cuda(current_grad, activation);

  // Convert gradients back to FP32 if needed
  if (config.use_fp16) {
    HalfPrecisionTraining::convert_to_fp32(layer_grad);
  }

  return layer_grad;
#else
  return backward(grad, activation, layer_idx);
#endif
}

std::vector<Matrix> &Transformer::parameters() {
  static std::vector<Matrix> all_params;
  all_params.clear();

  // Add embedding parameters
  all_params.push_back(token_embedding->get_embedding_table());

  // Add layer parameters
  for (auto &layer : layers) {
    // Add attention parameters
    all_params.push_back(layer->self_attention->query_proj);
    all_params.push_back(layer->self_attention->key_proj);
    all_params.push_back(layer->self_attention->value_proj);
    all_params.push_back(layer->self_attention->output_proj);

    // Add layer norm parameters - convert Vector to Matrix
    const Vector &gamma = layer->attention_ln->get_gamma();
    const Vector &beta = layer->attention_ln->get_beta();
    Matrix gamma_matrix(1, gamma.size());
    Matrix beta_matrix(1, beta.size());
    for (size_t i = 0; i < gamma.size(); ++i) {
      gamma_matrix(0, i) = gamma[i];
      beta_matrix(0, i) = beta[i];
    }
    all_params.push_back(gamma_matrix);
    all_params.push_back(beta_matrix);

    // Add feed forward parameters
    all_params.push_back(layer->feed_forward->w1);
    all_params.push_back(layer->feed_forward->w2);

    // Convert feed forward biases to matrices
    Matrix b1_matrix(1, layer->feed_forward->b1.size());
    Matrix b2_matrix(1, layer->feed_forward->b2.size());
    for (size_t i = 0; i < layer->feed_forward->b1.size(); ++i) {
      b1_matrix(0, i) = layer->feed_forward->b1[i];
    }
    for (size_t i = 0; i < layer->feed_forward->b2.size(); ++i) {
      b2_matrix(0, i) = layer->feed_forward->b2[i];
    }
    all_params.push_back(b1_matrix);
    all_params.push_back(b2_matrix);

    // Add final layer norm parameters
    const Vector &ffn_gamma = layer->ffn_ln->get_gamma();
    const Vector &ffn_beta = layer->ffn_ln->get_beta();
    Matrix ffn_gamma_matrix(1, ffn_gamma.size());
    Matrix ffn_beta_matrix(1, ffn_beta.size());
    for (size_t i = 0; i < ffn_gamma.size(); ++i) {
      ffn_gamma_matrix(0, i) = ffn_gamma[i];
      ffn_beta_matrix(0, i) = ffn_beta[i];
    }
    all_params.push_back(ffn_gamma_matrix);
    all_params.push_back(ffn_beta_matrix);
  }

  // Add final layer norm parameters
  const Vector &final_gamma = final_ln->get_gamma();
  const Vector &final_beta = final_ln->get_beta();
  Matrix final_gamma_matrix(1, final_gamma.size());
  Matrix final_beta_matrix(1, final_beta.size());
  for (size_t i = 0; i < final_gamma.size(); ++i) {
    final_gamma_matrix(0, i) = final_gamma[i];
    final_beta_matrix(0, i) = final_beta[i];
  }
  all_params.push_back(final_gamma_matrix);
  all_params.push_back(final_beta_matrix);

  return all_params;
}

void Transformer::save(std::ostream &os) const {
  // Save config
  os.write(reinterpret_cast<const char *>(&config), sizeof(config));

  // Save embeddings
  token_embedding->save(os);
  pos_encoding->save(os);

  // Save layers
  for (const auto &layer : layers) {
    layer->save(os);
  }

  // Save final layer norm
  final_ln->save(os);
}

void Transformer::load(std::istream &is) {
  // Load config
  is.read(reinterpret_cast<char *>(&config), sizeof(config));

  // Load embeddings
  token_embedding = TokenEmbedding::load(is);
  pos_encoding = PositionalEncoding::load(is);

  // Load layers
  layers.clear();
  for (size_t i = 0; i < config.num_layers; ++i) {
    auto layer = TransformerLayer::create(config);
    layer->load(is);
    layers.push_back(std::move(layer));
  }

  // Load final layer norm
  final_ln = LayerNorm::load(is);
}

Matrix TransformerLayer::backward(const Matrix &grad,
                                  const Matrix &input) const {
  std::cout << "TransformerLayer backward dimensions:" << std::endl;
  std::cout << "grad: " << grad.rows() << "x" << grad.cols() << std::endl;
  std::cout << "input: " << input.rows() << "x" << input.cols() << std::endl;

  // Verify dimensions match
  if (grad.rows() != input.rows() || grad.cols() != input.cols()) {
    throw std::runtime_error(
        "Gradient and input dimensions must match in layer backward. "
        "grad: " +
        std::to_string(grad.rows()) + "x" + std::to_string(grad.cols()) +
        " input: " + std::to_string(input.rows()) + "x" +
        std::to_string(input.cols()));
  }

  // Backward through feed forward
  Matrix d_residual2 = grad;
  Matrix normalized = ffn_ln->forward(input);
  Matrix d_ffn = feed_forward->backward(d_residual2, normalized);
  Matrix d_ln2 = ffn_ln->backward(d_ffn, input);

  // Backward through attention
  Matrix d_residual1 = d_ln2 + d_residual2;
  Matrix attn_normalized = attention_ln->forward(input);
  Matrix d_attn = self_attention->backward(d_residual1, attn_normalized);
  Matrix d_ln1 = attention_ln->backward(d_attn, input);

  return d_ln1;
}

Matrix Transformer::forward_cuda(const std::vector<int> &input_tokens,
                                 bool use_cache) {
#ifdef USE_CUDA
  if (!cuda_initialized) {
    throw std::runtime_error("CUDA not initialized");
  }

  // Pin the embedding table memory to prevent it from being paged out
  const Matrix &embedding_table = token_embedding->get_embedding_table();
  float *pinned_embedding;
  CUDA_CHECK(cudaMallocHost(&pinned_embedding, embedding_table.rows() *
                                                   embedding_table.cols() *
                                                   sizeof(float)));
  std::memcpy(pinned_embedding, embedding_table.data(),
              embedding_table.rows() * embedding_table.cols() * sizeof(float));

  // Allocate memory for embeddings using memory pool
  std::cout << "Allocating memory for embeddings" << std::endl;
  size_t embed_size = input_tokens.size() * config.hidden_size;
  float *embed_data = MemoryPool::allocate_static(embed_size * sizeof(float));
  Matrix embeddings(input_tokens.size(), config.hidden_size, embed_data);

  // Get embeddings using CUDA
  std::cout << "Getting embeddings using CUDA" << std::endl;
  token_embedding->forward_cuda(input_tokens, embeddings);
  std::cout << "Got embeddings" << std::endl;
  // Add positional encodings
  Matrix position_ids(input_tokens.size(), 1);
  for (size_t i = 0; i < input_tokens.size(); ++i) {
    position_ids(i, 0) = static_cast<float>(i);
  }
  // Compute positional encodings on GPU
  std::cout << "Computing positional encodings on GPU" << std::endl;
  Matrix pos_encodings = pos_encoding->forward(position_ids);

  // Add position encodings using CUDA
        // Add positional encodings using OpenMP
  #pragma omp parallel for
  for (size_t i = 0; i < embeddings.rows(); ++i) {
      for (size_t j = 0; j < embeddings.cols(); ++j) {
          embeddings(i, j) += pos_encodings(i % pos_encodings.rows(), j);
      }
  }
  // Create deep copy for hidden states
  hidden_states = Matrix(embeddings.rows(), embeddings.cols(), embeddings.data(), false);

  // Create attention mask if needed
  AttentionMask mask;
  if (!use_cache) {
    mask = AttentionMask::create_causal_mask(input_tokens.size());
  }

  for (size_t i = 0; i < layers.size(); ++i) {
    // Save activation for gradient checkpointing
    GradientCheckpoint::save_activation(hidden_states, i);

    // Forward through layer using CUDA
    hidden_states = layers[i]->forward(hidden_states, mask);

    // Convert to FP16 if enabled
    if (config.use_fp16) {
      HalfPrecisionTraining::convert_to_fp16(hidden_states);
    }
  }

  // Final layer normalization using CUDA
  hidden_states = final_ln->forward(hidden_states);

  // Instead of cublasSgemm, do the projection on CPU
  Matrix logits(hidden_states.rows(), config.vocab_size);

// CPU matrix multiplication
#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < logits.rows(); i++) {
    for (size_t j = 0; j < config.vocab_size; j++) {
      float sum = 0.0f;
      for (size_t k = 0; k < config.hidden_size; k++) {
        sum += hidden_states(i, k) * embedding_table(j, k);
      }
      logits(i, j) = sum;
    }
  }

  // Make a copy and cleanup
  Matrix result = logits;
  MemoryPool::deallocate_static(embed_data, embed_size * sizeof(float));

  return result;
#else
  throw std::runtime_error("CUDA support not enabled");
#endif
}

Matrix TransformerLayer::backward_cuda(const Matrix &grad,
                                       const Matrix &input) const {
#ifdef USE_CUDA
  throw std::runtime_error("CUDA implementation not available");
#else
  return backward(grad, input);
#endif
}

Transformer::~Transformer() {
  // Disable logging before CUDA cleanup
  Logger::getInstance().disableLogging();

  if (cuda_initialized) {
    cleanup_cuda();
    cuda_initialized = false;
  }
}

Transformer::Transformer(const Transformer &other) : config(other.config) {
  // Deep copy token embedding
  token_embedding = std::make_unique<TokenEmbedding>(*other.token_embedding);

  // Deep copy positional encoding
  pos_encoding = std::make_unique<PositionalEncoding>(*other.pos_encoding);

  // Deep copy layers
  layers.reserve(other.layers.size());
  for (const auto &layer : other.layers) {
    auto new_layer = std::make_unique<TransformerLayer>(*layer);
    layers.push_back(std::move(new_layer));
  }

  // Deep copy final layer norm
  final_ln = std::make_unique<LayerNorm>(*other.final_ln);

  // Deep copy language model head if it exists
  if (other.lm_head) {
    lm_head = std::make_unique<LanguageModelHead>(*other.lm_head);
  }
}

Transformer &Transformer::operator=(const Transformer &other) {
  if (this != &other) {
    config = other.config;

    // Deep copy token embedding
    token_embedding = std::make_unique<TokenEmbedding>(*other.token_embedding);

    // Deep copy positional encoding
    pos_encoding = std::make_unique<PositionalEncoding>(*other.pos_encoding);

    // Deep copy layers
    layers.clear();
    layers.reserve(other.layers.size());
    for (const auto &layer : other.layers) {
      layers.push_back(std::make_unique<TransformerLayer>(*layer));
    }

    // Deep copy final layer norm
    final_ln = std::make_unique<LayerNorm>(*other.final_ln);

    // Deep copy language model head if it exists
    if (other.lm_head) {
      lm_head = std::make_unique<LanguageModelHead>(*other.lm_head);
    } else {
      lm_head.reset();
    }
  }
  return *this;
}

void Transformer::backward(const Matrix &grad_output,
                           const std::vector<int> &input_tokens) {
  // Verify dimensions
  if (grad_output.cols() != config.hidden_size) {
    throw std::runtime_error("Gradient output dimension (" +
                             std::to_string(grad_output.cols()) +
                             ") must match hidden size (" +
                             std::to_string(config.hidden_size) + ")");
  }

  // Backpropagate through final layer norm
  Matrix grad = final_ln->backward(grad_output, hidden_states);
  std::cout << "outside of final ln with grad shape: " << grad.shape()
            << std::endl;
  // Backpropagate through transformer layers in reverse order
  for (int i = layers.size() - 1; i >= 0; --i) {
    Matrix cached_activation = GradientCheckpoint::get_activation(i);
    // Verify cached activation dimensions
    if (cached_activation.cols() != config.hidden_size) {
      throw std::runtime_error("Cached activation dimension mismatch");
    }
    std::cout << "layer " << i << " backward" << std::endl;
    grad = layers[i]->backward(grad, cached_activation);
  }

  // Backpropagate through embeddings
  if (grad.rows() != input_tokens.size()) {
    throw std::runtime_error("Gradient rows (" + std::to_string(grad.rows()) +
                             ") must match sequence length (" +
                             std::to_string(input_tokens.size()) + ")");
  }

  // Update token embeddings
  token_embedding->backward(grad, input_tokens);
}

// Add member to store last hidden states for backward pass
Matrix hidden_states;