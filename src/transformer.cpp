#include "../include/transformer.hpp"
#include "../include/cuda/cublas_check.cuh"
#include "../include/cuda/cuda_check.cuh"
#include "../include/logger.hpp"
#include "../include/utils.hpp"
#include <cublas_v2.h>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <stdexcept>

extern cublasHandle_t cublas_handle;

// TransformerConfig implementation
TransformerConfig::TransformerConfig(size_t vocab_size, size_t max_seq_length,
                                     size_t hidden_size, size_t num_layers,
                                     size_t num_heads, size_t batch_size,
                                     size_t num_epochs)
    : vocab_size(vocab_size), max_seq_length(max_seq_length),
      hidden_size(hidden_size), num_layers(num_layers), num_heads(num_heads),
      head_dim(hidden_size / num_heads), intermediate_size(4 * hidden_size),
      dropout_prob(0.1f), use_flash_attention(true), use_rope(true),
      use_sliding_window(false), window_size(512), use_gqa(false),
      num_kv_heads(num_heads / 2), use_fp16(false),
      use_gradient_checkpointing(true), memory_pool_size(1024),
      batch_size(batch_size), num_epochs(num_epochs), dropout_rate(0.1f),
      weight_decay(0.01f), paths{
                               "models",            // save_directory
                               "transformer_model", // model_name
                               2                    // checkpoint_frequency
                           } {
  std::cout << "entering TransformerConfig constructor" << std::endl;
  if (hidden_size % num_heads != 0) {
    throw std::invalid_argument(
        "Hidden size must be divisible by number of heads");
  }
  std::cout << "exiting TransformerConfig constructor" << std::endl;
}

// TransformerLayer implementation
TransformerLayer::TransformerLayer(const TransformerConfig &config_, size_t idx)
    : config(config_), layer_idx(idx) {
  // Initialize components
  std::cout << "Initializing TransformerLayer " << idx
            << " with GQA config:" << std::endl;
  std::cout << "- use_gqa: " << (config.use_gqa ? "true" : "false")
            << std::endl;
  std::cout << "- num_kv_heads: " << config.num_kv_heads << std::endl;

  self_attention = std::make_unique<MultiHeadAttention>(
      config.hidden_size, config.num_heads, config.head_dim,
      config.dropout_prob, config.use_flash_attention, config.use_rope,
      config.use_sliding_window, config.window_size, config.use_gqa,
      config.num_kv_heads, config.max_seq_length);

  attention_ln = std::make_unique<LayerNorm>(config.hidden_size);
  feed_forward = std::make_unique<FeedForward>(config.hidden_size,
                                               config.intermediate_size);
  ffn_ln = std::make_unique<LayerNorm>(config.hidden_size);

  // Initialize dropout layers
  attention_dropout = std::make_unique<Dropout>(config.dropout_rate);
  ffn_dropout = std::make_unique<Dropout>(config.dropout_rate);
}

Matrix TransformerLayer::forward(const Matrix &input, const AttentionMask &mask,
                                 const std::optional<KVCache> &kv_cache) {
  std::cout << "=== TransformerLayer::forward START ===" << std::endl;

  // Layer norm before attention
  Matrix normalized = attention_ln->forward(input);

  // Cache the normalized input for attention backward pass
  std::string attn_key = "attn_norm_" + std::to_string(layer_idx);
  GradientCheckpoint::cache_activation(attn_key, normalized);

  // Self attention
  Matrix attention_output = self_attention->forward(normalized, mask, kv_cache);
  std::cout << "attention output: " << attention_output.rows() << "x"
            << attention_output.cols() << std::endl;
  if (training) {
    std::cout << "attention dropout" << std::endl;
    attention_output = attention_dropout->forward(attention_output, true);
  }
  std::cout << "calculating residual" << std::endl;
  Matrix residual = attention_output + normalized;
  std::cout << "calculating attention ln" << std::endl;
  Matrix norm1 = attention_ln->forward(residual);

  // Cache the normalized input for feed forward backward pass
  std::string ffn_key = "ffn_norm_" + std::to_string(layer_idx);
  GradientCheckpoint::cache_activation(ffn_key, norm1);

  // Feed forward
  Matrix ff_output = feed_forward->forward(norm1);
  if (training) {
    ff_output = ffn_dropout->forward(ff_output, true);
  }
  residual = ff_output + norm1;

  return ffn_ln->forward(residual);
}

Matrix TransformerLayer::backward(const Matrix &grad_output,
                                  const Matrix &input,
                                  const Matrix &target_distribution) {
  std::cout << "=== TransformerLayer::backward START ===" << std::endl;
  std::cout << "Grad output dimensions: " << grad_output.rows() << "x"
            << grad_output.cols() << std::endl;
  std::cout << "Input dimensions: " << input.rows() << "x" << input.cols()
            << std::endl;

  try {
    // Get the cached normalized input for feed forward
    std::string ffn_key = "ffn_norm_" + std::to_string(layer_idx);
    Matrix ffn_normalized = GradientCheckpoint::get_activation(ffn_key);

    // Backward through feed forward network
    Matrix ff_dropout_grad =
        training ? ffn_dropout->backward(grad_output) : grad_output;
    Matrix ffn_grad = feed_forward->backward(ff_dropout_grad, ffn_normalized);
    std::cout << "FFN grad dimensions: " << ffn_grad.rows() << "x"
              << ffn_grad.cols() << std::endl;

    // Backward through feed forward layer norm
    Matrix ffn_ln_grad = ffn_ln->backward(ffn_grad, input);
    std::cout << "FFN LN grad dimensions: " << ffn_ln_grad.rows() << "x"
              << ffn_ln_grad.cols() << std::endl;

    // Check dimensions before first residual addition
    std::cout << "About to add FFN LN grad (" << ffn_ln_grad.rows() << "x"
              << ffn_ln_grad.cols() << ") with grad_output ("
              << grad_output.rows() << "x" << grad_output.cols() << ")"
              << std::endl;

    if (ffn_ln_grad.rows() != grad_output.rows() ||
        ffn_ln_grad.cols() != grad_output.cols()) {
      throw std::runtime_error(
          "Dimension mismatch in FFN residual: ffn_ln_grad(" +
          std::to_string(ffn_ln_grad.rows()) + "," +
          std::to_string(ffn_ln_grad.cols()) + ") != grad_output(" +
          std::to_string(grad_output.rows()) + "," +
          std::to_string(grad_output.cols()) + ")");
    }

    // First residual addition
    Matrix residual_grad = ffn_ln_grad + grad_output;
    std::cout << "Residual grad dimensions after first addition: "
              << residual_grad.rows() << "x" << residual_grad.cols()
              << std::endl;

    // Get the cached normalized input for attention
    std::string attn_key = "attn_norm_" + std::to_string(layer_idx);
    Matrix attn_normalized = GradientCheckpoint::get_activation(attn_key);

    // Backward through self attention
    Matrix attn_dropout_grad =
        training ? attention_dropout->backward(residual_grad) : residual_grad;
    Matrix attention_grad = self_attention->backward(
        attn_dropout_grad, attn_normalized, target_distribution);
    std::cout << "Attention grad dimensions: " << attention_grad.rows() << "x"
              << attention_grad.cols() << std::endl;

    // Backward through attention layer norm
    Matrix attention_ln_grad = attention_ln->backward(attention_grad, input);
    std::cout << "Attention LN grad dimensions: " << attention_ln_grad.rows()
              << "x" << attention_ln_grad.cols() << std::endl;

    // Check dimensions before second residual addition
    std::cout << "About to add attention LN grad (" << attention_ln_grad.rows()
              << "x" << attention_ln_grad.cols() << ") with residual_grad ("
              << residual_grad.rows() << "x" << residual_grad.cols() << ")"
              << std::endl;

    if (attention_ln_grad.rows() != residual_grad.rows() ||
        attention_ln_grad.cols() != residual_grad.cols()) {
      throw std::runtime_error(
          "Dimension mismatch in attention residual: attention_ln_grad(" +
          std::to_string(attention_ln_grad.rows()) + "," +
          std::to_string(attention_ln_grad.cols()) + ") != residual_grad(" +
          std::to_string(residual_grad.rows()) + "," +
          std::to_string(residual_grad.cols()) + ")");
    }

    // Second residual addition
    Matrix final_grad = attention_ln_grad + residual_grad;
    std::cout << "Final grad dimensions after second addition: "
              << final_grad.rows() << "x" << final_grad.cols() << std::endl;

    std::cout << "=== TransformerLayer::backward END ===" << std::endl;
    return final_grad;

  } catch (const std::exception &e) {
    std::cerr << "Error in TransformerLayer::backward: " << e.what()
              << std::endl;
    throw;
  }
}

// Transformer implementation
Transformer::Transformer(const TransformerConfig& config, std::unique_ptr<SAM> sam_optimizer) 
    : config(config), optimizer(std::move(sam_optimizer)) {
    std::cout << "\n=== Transformer::constructor START ===" << std::endl;

    // Initialize token embedding
    token_embedding = std::make_unique<TokenEmbedding>(config.vocab_size, config.hidden_size);

    // Initialize positional encoding
    pos_encoding = std::make_unique<PositionalEncoding>(config.max_seq_length,
                                                        config.hidden_size);

    // Initialize transformer layers
    layers.reserve(config.num_layers);
    m_kv_caches.reserve(config.num_layers);
    for (size_t i = 0; i < config.num_layers; ++i) {
        layers.push_back(std::make_unique<TransformerLayer>(config, i));
        m_kv_caches.emplace_back(config.max_seq_length);
    }

    // Initialize final layer normalization
    final_ln = std::make_unique<LayerNorm>(config.hidden_size);

    // Initialize the language model head
    lm_head = std::make_unique<LanguageModelHead>(config.hidden_size,
                                                  config.vocab_size);

    // Initialize tokenizer
    tokenizer = std::make_unique<Tokenizer>();

    // Create default optimizer if none provided
    if (!optimizer) {
        optimizer = std::make_unique<SAM>(0.05f);
    }

    std::cout << "=== Transformer::constructor END ===\n" << std::endl;
}

Matrix Transformer::forward(const std::vector<std::vector<int>> &batch_tokens, bool use_cache) {
    std::cout << "\n=== Transformer::forward START ===" << std::endl;
    
    // Calculate max sequence length in batch
    size_t max_seq_len = 0;
    for (const auto& tokens : batch_tokens) {
        max_seq_len = std::max(max_seq_len, tokens.size());
    }
    
    // Create batched input matrix
    Matrix embeddings(batch_tokens.size() * max_seq_len, config.hidden_size, 0.0f);
    // Process all sequences in one batch
    Matrix all_embeddings = token_embedding->forward(batch_tokens);
    embeddings = all_embeddings;  // The embeddings will already be in the correct shape

    std::cout << "before position ids" << std::endl;
    Matrix position_ids(max_seq_len, 1);
    for (size_t i = 0; i < max_seq_len; ++i) {
        position_ids(i, 0) = static_cast<float>(i);
    }
    std::cout << "after position ids" << std::endl;
    Matrix pos_encodings = pos_encoding->forward(position_ids);
   
    // Replicate positional encodings for each item in the batch
    Matrix batched_pos_encodings(embeddings.rows(), embeddings.cols(), 0.0f);
    for (size_t b = 0; b < batch_tokens.size(); b++) {
        for (size_t i = 0; i < max_seq_len; i++) {
            for (size_t j = 0; j < config.hidden_size; j++) {
                batched_pos_encodings(b * max_seq_len + i, j) = pos_encodings(i, j);
            }
        }
    }
    std::cout << "after pos encoding forward" << std::endl;
    embeddings += batched_pos_encodings;  // Now dimensions match
    std::cout << "after embeddings + pos encodings" << std::endl;
    // Create causal mask for next-token prediction
    std::cout << "before mask" << std::endl;
    AttentionMask mask = AttentionMask::create_causal_mask(max_seq_len);
    std::cout << "after mask" << std::endl;
    std::cout << "before hidden states" << std::endl;
    // Forward through layers
    hidden_states = embeddings;
    std::cout << "after hidden states" << std::endl;
    std::vector<Matrix> activations;
    activations.reserve(layers.size());
    std::cout << "before activations" << std::endl;
    for (size_t i = 0; i < layers.size(); ++i) {
        activations.push_back(hidden_states);
        hidden_states = layers[i]->forward(
            hidden_states, mask,
            use_cache ? std::optional<KVCache>(m_kv_caches[i]) : std::nullopt);
    }
    std::cout << "after activations" << std::endl;
    // Final layer normalization
    std::cout << "before final ln" << std::endl;
    hidden_states = final_ln->forward(hidden_states);
    std::cout << "after final ln" << std::endl;
    // Store activations for backward pass
    std::cout << "before last hidden states" << std::endl;
    last_hidden_states = hidden_states;
    std::cout << "after last hidden states" << std::endl;
    m_layer_activations = std::move(activations);
    std::cout << "after layer activations" << std::endl;

    std::cout << "=== Transformer::forward END ===\n" << std::endl;
    return hidden_states;
}

void Transformer::clear_kv_cache() {
  for (auto &cache : m_kv_caches) {
    cache.clear();
  }
}

void Transformer::backward(const Matrix &grad_output, const std::vector<std::vector<int>> &batch_tokens,
                           float learning_rate) {
  std::cout << "\n=== Transformer::backward START ===" << std::endl;
  std::cout << "before grad output" << std::endl;

  Matrix current_grad = grad_output;
  std::cout << "Initial grad dimensions: " << current_grad.rows() << "x"
            << current_grad.cols() << std::endl;

  // Backward through final layer norm
  const Matrix &last_activation = m_layer_activations.back();
  std::cout << "Last activation dimensions: " << last_activation.rows() << "x"
            << last_activation.cols() << std::endl;

  if (current_grad.rows() != last_activation.rows() ||
      current_grad.cols() != last_activation.cols()) {
        std::cout << "dimension mismatch in final layer norm!!!" << std::endl;
    throw std::runtime_error(
        "Dimension mismatch in final layer norm backward: grad(" +
        std::to_string(current_grad.rows()) + "," +
        std::to_string(current_grad.cols()) + ") != activation(" +
        std::to_string(last_activation.rows()) + "," +
        std::to_string(last_activation.cols()) + ")");
  }
  std::cout << "after final ln" << std::endl;
  current_grad = final_ln->backward(current_grad, last_activation);
  std::cout << "After final LN grad dimensions: " << current_grad.rows() << "x"
            << current_grad.cols() << std::endl;

  // Backward through layers in reverse order
  for (int i = layers.size() - 1; i >= 0; --i) {
    std::cout << "\nProcessing layer " << i << std::endl;
    const Matrix &layer_input = m_layer_activations[i];
    std::cout << "Layer input dimensions: " << layer_input.rows() << "x"
              << layer_input.cols() << std::endl;
    std::cout << "Current grad dimensions: " << current_grad.rows() << "x"
              << current_grad.cols() << std::endl;

    if (current_grad.rows() != layer_input.rows() ||
        current_grad.cols() != layer_input.cols()) {
      throw std::runtime_error(
          "Dimension mismatch in layer " + std::to_string(i) +
          " backward: grad(" + std::to_string(current_grad.rows()) + "," +
          std::to_string(current_grad.cols()) + ") != input(" +
          std::to_string(layer_input.rows()) + "," +
          std::to_string(layer_input.cols()) + ")");
    }

    current_grad = layers[i]->backward(current_grad, layer_input, Matrix());
    std::cout << "After layer " << i
              << " grad dimensions: " << current_grad.rows() << "x"
              << current_grad.cols() << std::endl;
  }

  // Get parameters and their gradients
  auto& params = parameters();
  auto& grads = parameter_gradients();
  
  // Convert params to vector of pointers for SAM
  std::vector<Matrix*> param_ptrs;
  for (auto& param : params) {
      param_ptrs.push_back(&param);
  }
  
  // First step of SAM
  optimizer->first_step(param_ptrs, grads);
  std::cout << "after first step" << std::endl;
  // Recompute gradients at the perturbed point
  Matrix perturbed_output = forward(batch_tokens);
  std::cout << "after perturbed output" << std::endl;
  Matrix perturbed_grad = compute_loss_gradients(perturbed_output, batch_tokens);
  std::cout << "after perturbed grad" << std::endl;
  auto& perturbed_grads = parameter_gradients();
  std::cout << "after perturbed grads" << std::endl;
  
  // Second step of SAM
  optimizer->second_step(param_ptrs, perturbed_grads);

  std::cout << "=== Transformer::backward END ===\n" << std::endl;
}

void Transformer::update_parameters(float learning_rate) {
  std::cout << "=== Transformer::update_parameters START ===" << std::endl;

  // Update Matrix parameters
  auto &params = parameters();
  auto &grads = parameter_gradients();

  std::cout << "Number of matrix parameters: " << params.size() << std::endl;
  std::cout << "Number of matrix gradients: " << grads.size() << std::endl;

  for (size_t i = 0; i < params.size(); ++i) {
    Matrix &param = params[i];
    const Matrix &grad = grads[i];

    std::cout << "Updating matrix parameter " << i << ": ";
    std::cout << "param shape=" << param.rows() << "x" << param.cols()
              << ", grad shape=" << grad.rows() << "x" << grad.cols()
              << std::endl;

    if (param.rows() != grad.rows() || param.cols() != grad.cols()) {
      throw std::runtime_error("Dimension mismatch in matrix update: param(" +
                               std::to_string(param.rows()) + "," +
                               std::to_string(param.cols()) + ") != grad(" +
                               std::to_string(grad.rows()) + "," +
                               std::to_string(grad.cols()) + ")");
    }

    // Update rule: param = param - learning_rate * grad
    for (size_t j = 0; j < param.size(); ++j) {
      param.data()[j] -= learning_rate * grad.data()[j];
    }
  }

  // Update Vector parameters for each layer
  for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
    std::cout << "\nProcessing layer " << layer_idx << std::endl;
    auto &layer = layers[layer_idx];

    // Update attention parameters
    auto &attn_params = layer->self_attention->parameters();
    auto &attn_grads = layer->self_attention->parameter_gradients();

    std::cout << "Attention vectors: " << attn_params.vectors.size()
              << " parameters, " << attn_grads.vectors.size() << " gradients"
              << std::endl;

    // Update attention biases using computed gradients
    for (size_t i = 0; i < attn_params.vectors.size(); ++i) {
      auto &bias = attn_params.vectors[i];
      const auto &bias_grad = attn_grads.vectors[i];

      std::cout << "Attention bias " << i << ": bias size=" << bias.get().size()
                << ", grad size=" << bias_grad.get().size() << std::endl;

      if (bias.get().size() != bias_grad.get().size()) {
        throw std::runtime_error("Dimension mismatch in attention bias update");
      }

      for (size_t j = 0; j < bias.get().size(); ++j) {
        bias.get().data()[j] -= learning_rate * bias_grad.get().data()[j];
      }
    }

    // Update layer norm parameters
    auto &ln_params = layer->attention_ln->parameters();
    auto &ln_grads = layer->attention_ln->parameter_gradients();

    std::cout << "Layer norm vectors: " << ln_params.size() << " parameters, "
              << ln_grads.size() << " gradients" << std::endl;

    for (size_t i = 0; i < ln_params.size(); ++i) {
      auto &param = ln_params[i];
      const auto &grad = ln_grads[i];

      std::cout << "Layer norm param " << i
                << ": param size=" << param.get().size()
                << ", grad size=" << grad.get().size() << std::endl;

      if (param.get().size() != grad.get().size()) {
        throw std::runtime_error("Dimension mismatch in layer norm update");
      }

      for (size_t j = 0; j < param.get().size(); ++j) {
        param.get().data()[j] -= learning_rate * grad.get().data()[j];
      }
    }

    // Update feed forward parameters
    auto &ffn_params = layer->feed_forward->parameters();
    auto &ffn_grads = layer->feed_forward->parameter_gradients();

    std::cout << "Feed forward vectors: " << ffn_params.vectors.size()
              << " parameters, " << ffn_grads.vectors.size() << " gradients"
              << std::endl;

    for (size_t i = 0; i < ffn_params.vectors.size(); ++i) {
      auto &bias = ffn_params.vectors[i];
      const auto &bias_grad = ffn_grads.vectors[i];

      std::cout << "Feed forward bias " << i
                << ": bias size=" << bias.get().size()
                << ", grad size=" << bias_grad.get().size() << std::endl;

      if (bias.get().size() != bias_grad.get().size()) {
        throw std::runtime_error(
            "Dimension mismatch in feed forward bias update");
      }

      for (size_t j = 0; j < bias.get().size(); ++j) {
        bias.get().data()[j] -= learning_rate * bias_grad.get().data()[j];
      }
    }
  }

  // Clear gradients after update
  parameter_grads.reset();

  std::cout << "=== Transformer::update_parameters END ===" << std::endl;
}

std::vector<Matrix> &Transformer::parameters() {
  static std::vector<Matrix> all_params;
  all_params.clear();

  // Token embedding parameters (only Matrix)
  if (token_embedding) {
    auto &token_params = token_embedding->parameters();
    for (const auto &param : token_params) {
      all_params.push_back(param.get());
    }
  }

  // Layer parameters
  for (const auto &layer : layers) {
    // Attention parameters
    auto &attention_params = layer->self_attention->parameters();
    for (const auto &param : attention_params.matrices) {
      all_params.push_back(param.get());
    }

    // Layer norm parameters (only Vectors, skip)

    // Feed forward parameters
    auto &ffn_params = layer->feed_forward->parameters();
    for (const auto &param : ffn_params.matrices) {
      all_params.push_back(param.get());
    }
  }

  return all_params;
}

Transformer::~Transformer() {
  std::cout << "Transformer destructor called" << std::endl;
}

void Transformer::train(const std::vector<std::pair<std::string, std::string>>& training_data,
                       const std::vector<std::pair<std::string, std::string>>& validation_data,
                       size_t num_epochs, float learning_rate,
                       std::function<void(size_t)> checkpoint_callback) {
    const size_t validate_every = 100;
    size_t step = 0;
    float total_loss = 0.0f;
    size_t batch_count = 0;
    
    std::cout << "Starting training with " << training_data.size() << " samples\n";
    std::cout << "Config batch size: " << config.batch_size << "\n";
    
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "\nEpoch " << epoch + 1 << "/" << num_epochs << "\n";
        float epoch_loss = 0.0f;
        size_t epoch_batch_count = 0;

        // Group data into batches
        for (size_t batch_start = 0; batch_start < training_data.size(); batch_start += config.batch_size) {
            size_t batch_end = std::min(batch_start + config.batch_size, training_data.size());
            std::vector<std::vector<int>> batch_inputs;
            std::vector<std::vector<int>> batch_targets;
            
            // Collect batch_size samples
            for (size_t i = batch_start; i < batch_end; i++) {
                const auto& [input_text, target_text] = training_data[i];
                batch_inputs.push_back(tokenizer->encode(input_text));
                batch_targets.push_back(tokenizer->encode(target_text));
            }
            
            optimizer->zero_grad();
            
            Matrix output = forward(batch_inputs);
            std::cout << "output shape: " << output.rows() << "x" << output.cols() << std::endl;
            std::cout << "hello govna!" << std::endl;

            // Create batched target distribution
            size_t max_seq_len = output.rows() / batch_inputs.size();
            Matrix target_distribution(output.rows(), config.vocab_size, 0.0f);
            std::cout << "target distribution shape: " << target_distribution.rows() << "x" << target_distribution.cols() << std::endl;
            for (size_t b = 0; b < batch_inputs.size(); b++) {
                if (batch_targets[b].size() > max_seq_len) {
                    std::cout << "Warning: Truncating sequence " << b << " from length " 
                              << batch_targets[b].size() << " to " << max_seq_len << std::endl;
                }
                for (size_t i = 0; i < std::min(batch_targets[b].size(), max_seq_len); i++) {
                    if (batch_targets[b][i] >= 0 && batch_targets[b][i] < config.vocab_size) {
                        target_distribution(b * max_seq_len + i, batch_targets[b][i]) = 1.0f;
                    }
                }
            }
            std::cout << "target distribution shape afterwards: " << target_distribution.rows() << "x" << target_distribution.cols() << std::endl;
            float batch_loss = Utils::compute_batch_loss(output, target_distribution);
            std::cout << "Batch " << batch_count + 1 << "/" << training_data.size() 
                     << ", Loss: " << batch_loss << "\n";
            std::cout << "batch loss: " << batch_loss << std::endl;
            
            epoch_loss += batch_loss;
            total_loss += batch_loss;
            ++epoch_batch_count;
            ++batch_count;
            
            std::cout << "Starting backward pass...\n";
            backward(output, batch_inputs, learning_rate);
            std::cout << "Backward pass complete\n";
            
            if (++step % validate_every == 0) {
                std::cout << "\nStarting validation...\n";
                float val_loss = Utils::evaluate_validation(*this, *tokenizer, validation_data);
                std::cout << "Epoch " << epoch + 1 << ", Step " << step 
                         << ", Validation loss: " << val_loss << std::endl;
            }
            
            checkpoint_callback(step);
            std::cout << "Batch complete\n";
        }
        
        float avg_epoch_loss = epoch_loss / epoch_batch_count;
        float avg_total_loss = total_loss / batch_count;
        std::cout << "\nEpoch " << epoch + 1 << " Summary:\n";
        std::cout << "Average Epoch Loss: " << avg_epoch_loss << "\n";
        std::cout << "Average Total Loss: " << avg_total_loss << "\n";
    }
    
    std::cout << "Final Average Loss: " << (total_loss / batch_count) << "\n";
}

Matrix Transformer::compute_loss_gradients(const Matrix& logits, const std::vector<std::vector<int>>& batch_targets) {
    Matrix loss_grad(logits.rows(), logits.cols(), 0.0f);
    
    size_t batch_size = batch_targets.size();
    size_t seq_len = logits.rows() / batch_size;
    
    // For each batch and sequence position
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < seq_len; i++) {
            size_t row = b * seq_len + i;
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < logits.cols(); j++) {
                max_val = std::max(max_val, logits(row, j));
            }
            
            // Compute softmax denominator
            float sum_exp = 0.0f;
            std::vector<float> exp_vals(logits.cols());
            for (size_t j = 0; j < logits.cols(); j++) {
                exp_vals[j] = std::exp(logits(row, j) - max_val);
                sum_exp += exp_vals[j];
            }
            
            // Compute gradients
            for (size_t j = 0; j < logits.cols(); j++) {
                float softmax_out = exp_vals[j] / sum_exp;
                // If this position has a target token, compute cross-entropy gradient
                if (i < batch_targets[b].size()) {
                    loss_grad(row, j) = softmax_out - (j == static_cast<size_t>(batch_targets[b][i]) ? 1.0f : 0.0f);
                }
            }
        }
    }
    
    // Scale gradients by sequence length for better numerical stability
    float scale = 1.0f / static_cast<float>(batch_targets.size());
    for (size_t i = 0; i < loss_grad.size(); i++) {
        loss_grad.data()[i] *= scale;
    }
    
    return loss_grad;
}

void Transformer::save_model(const std::string &path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file for saving model: " + path);
    }

    // Save token embedding
    token_embedding->save(ofs);

    // Save positional encoding
    pos_encoding->save(ofs);

    // Save transformer layers
    for (const auto &layer : layers) {
        layer->save(ofs);
    }

    // Save final layer normalization
    final_ln->save(ofs);

    // Save language model head
    lm_head->save(ofs);

    ofs.close();
}