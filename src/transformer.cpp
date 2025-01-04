#include "../include/transformer.hpp"
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
                                     size_t num_heads, size_t batch_size,
                                     size_t num_epochs)
    : vocab_size(vocab_size), max_seq_length(max_seq_length),
      hidden_size(hidden_size), num_layers(num_layers), num_heads(num_heads),
      head_dim(hidden_size / num_heads), intermediate_size(4 * hidden_size),
      dropout_prob(0.1f), use_flash_attention(true), use_rope(true),
      use_sliding_window(false), window_size(512), use_gqa(false),
      num_kv_heads(num_heads), use_cuda(true), batch_size(batch_size),
      num_epochs(num_epochs) {
  std::cout << "entering TransformerConfig constructor" << std::endl;
  if (hidden_size % num_heads != 0) {
    throw std::invalid_argument(
        "Hidden size must be divisible by number of heads");
  }
  std::cout << "exiting TransformerConfig constructor" << std::endl;
}

// TransformerLayer implementation
TransformerLayer::TransformerLayer(const TransformerConfig &config, size_t idx)
    : kv_cache(config.max_seq_length), 
      config(config),
      layer_idx(idx) {
  std::cout << "entering TransformerLayer constructor" << std::endl;
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
  std::cout << "exiting TransformerLayer constructor" << std::endl;
}

Matrix TransformerLayer::forward(const Matrix &input, const AttentionMask &mask,
                               const std::optional<KVCache> &kv_cache) {
    std::cout << "=== TransformerLayer::forward START ===" << std::endl;
    std::cout << "Input matrix shape: " << input.rows() << "x" << input.cols() << std::endl;
    
    // Layer norm before attention
    std::cout << "Applying attention layer normalization..." << std::endl;
    Matrix normalized = attention_ln->forward(input);
    std::cout << "Normalized matrix shape: " << normalized.rows() << "x" << normalized.cols() << std::endl;
    
    // Cache the normalized input for backward pass
    std::cout << "Caching normalized input for layer " << layer_idx << std::endl;
    GradientCheckpoint::cache_activation(std::to_string(layer_idx), normalized);
    
    // Self attention
    std::cout << "Applying self attention..." << std::endl;
    Matrix attention_output = self_attention->forward(normalized, mask, kv_cache);
    std::cout << "Attention output shape: " << attention_output.rows() << "x" << attention_output.cols() << std::endl;
    
    // Scale residual connection to prevent value explosion
    const float residual_scale = 0.5f;
    std::cout << "Scaling residual connection with factor " << residual_scale << std::endl;
    for(size_t i = 0; i < attention_output.size(); i++) {
        attention_output.data()[i] *= residual_scale;
    }
    
    // Check dimensions before first residual connection
    std::cout << "About to add attention output (" << attention_output.rows() << "x" << attention_output.cols() 
              << ") with input (" << input.rows() << "x" << input.cols() << ")" << std::endl;
    
    if (attention_output.rows() != input.rows() || attention_output.cols() != input.cols()) {
        throw std::runtime_error("Dimension mismatch in first residual connection: attention_output(" + 
                               std::to_string(attention_output.rows()) + "," + 
                               std::to_string(attention_output.cols()) + ") != input(" +
                               std::to_string(input.rows()) + "," +
                               std::to_string(input.cols()) + ")");
    }
    
    // Add first residual connection
    std::cout << "Adding first residual connection..." << std::endl;
    Matrix residual1 = attention_output + input;
    std::cout << "First residual shape: " << residual1.rows() << "x" << residual1.cols() << std::endl;
    
    // Layer norm before feed forward
    std::cout << "Applying feed forward layer normalization..." << std::endl;
    Matrix ffn_normalized = ffn_ln->forward(residual1);
    std::cout << "FFN normalized shape: " << ffn_normalized.rows() << "x" << ffn_normalized.cols() << std::endl;
    
    // Cache the normalized input for feed forward backward pass
    std::cout << "Caching FFN normalized input for layer " << layer_idx << std::endl;
    GradientCheckpoint::cache_activation(std::to_string(layer_idx) + "_ffn", ffn_normalized);
    std::cout << "Cached FFN activation successfully" << std::endl;
    
    // Feed forward
    std::cout << "Applying feed forward network..." << std::endl;
    Matrix ffn_output = feed_forward->forward(ffn_normalized);
    std::cout << "FFN output shape: " << ffn_output.rows() << "x" << ffn_output.cols() << std::endl;
    
    // Scale second residual connection
    std::cout << "Scaling second residual connection..." << std::endl;
    for(size_t i = 0; i < ffn_output.size(); i++) {
        ffn_output.data()[i] *= residual_scale;
    }
    std::cout << "Scaled FFN output" << std::endl;
    
    // Check dimensions before second residual connection
    std::cout << "About to add FFN output (" << ffn_output.rows() << "x" << ffn_output.cols() 
              << ") with residual1 (" << residual1.rows() << "x" << residual1.cols() << ")" << std::endl;
    
    if (ffn_output.rows() != residual1.rows() || ffn_output.cols() != residual1.cols()) {
        throw std::runtime_error("Dimension mismatch in second residual connection: ffn_output(" + 
                               std::to_string(ffn_output.rows()) + "," + 
                               std::to_string(ffn_output.cols()) + ") != residual1(" +
                               std::to_string(residual1.rows()) + "," +
                               std::to_string(residual1.cols()) + ")");
    }
    
    std::cout << "Adding second residual connection..." << std::endl;
    Matrix residual2 = ffn_output + residual1;
    std::cout << "Second residual shape: " << residual2.rows() << "x" << residual2.cols() << std::endl;
    
    std::cout << "=== TransformerLayer::forward END ===" << std::endl;
    return residual2;
}

Matrix TransformerLayer::backward(const Matrix &grad_output, const Matrix &input, const Matrix &target_distribution) {
    std::cout << "=== TransformerLayer::backward START ===" << std::endl;
    std::cout << "Grad output dimensions: " << grad_output.rows() << "x" << grad_output.cols() << std::endl;
    std::cout << "Input dimensions: " << input.rows() << "x" << input.cols() << std::endl;
    
    try {
        // Get the cached normalized input for feed forward
        Matrix ffn_normalized = GradientCheckpoint::get_activation(std::to_string(layer_idx) + "_ffn");
        std::cout << "FFN normalized dimensions: " << ffn_normalized.rows() << "x" << ffn_normalized.cols() << std::endl;
        
        // Backward through feed forward network
        Matrix ffn_grad = feed_forward->backward(grad_output, ffn_normalized);
        std::cout << "FFN grad dimensions: " << ffn_grad.rows() << "x" << ffn_grad.cols() << std::endl;
        
        // Backward through feed forward layer norm
        Matrix ffn_ln_grad = ffn_ln->backward(ffn_grad, input);
        std::cout << "FFN LN grad dimensions: " << ffn_ln_grad.rows() << "x" << ffn_ln_grad.cols() << std::endl;
        
        // Check dimensions before first residual addition
        std::cout << "About to add FFN LN grad (" << ffn_ln_grad.rows() << "x" << ffn_ln_grad.cols() 
                 << ") with grad_output (" << grad_output.rows() << "x" << grad_output.cols() << ")" << std::endl;
        
        if (ffn_ln_grad.rows() != grad_output.rows() || ffn_ln_grad.cols() != grad_output.cols()) {
            throw std::runtime_error("Dimension mismatch in FFN residual: ffn_ln_grad(" + 
                                   std::to_string(ffn_ln_grad.rows()) + "," + 
                                   std::to_string(ffn_ln_grad.cols()) + ") != grad_output(" +
                                   std::to_string(grad_output.rows()) + "," +
                                   std::to_string(grad_output.cols()) + ")");
        }
        
        // First residual addition
        Matrix residual_grad = ffn_ln_grad + grad_output;
        std::cout << "Residual grad dimensions after first addition: " << residual_grad.rows() << "x" << residual_grad.cols() << std::endl;
        
        // Get the cached normalized input for attention
        Matrix attn_normalized = GradientCheckpoint::get_activation(std::to_string(layer_idx));
        std::cout << "Attention normalized dimensions: " << attn_normalized.rows() << "x" << attn_normalized.cols() << std::endl;
        
        // Backward through self attention with the normalized input
        Matrix attention_grad = self_attention->backward(residual_grad, attn_normalized, target_distribution);
        std::cout << "Attention grad dimensions: " << attention_grad.rows() << "x" << attention_grad.cols() << std::endl;
        
        // Backward through attention layer norm
        Matrix attention_ln_grad = attention_ln->backward(attention_grad, input);
        std::cout << "Attention LN grad dimensions: " << attention_ln_grad.rows() << "x" << attention_ln_grad.cols() << std::endl;
        
        // Check dimensions before second residual addition
        std::cout << "About to add attention LN grad (" << attention_ln_grad.rows() << "x" << attention_ln_grad.cols() 
                 << ") with residual_grad (" << residual_grad.rows() << "x" << residual_grad.cols() << ")" << std::endl;
        
        if (attention_ln_grad.rows() != residual_grad.rows() || attention_ln_grad.cols() != residual_grad.cols()) {
            throw std::runtime_error("Dimension mismatch in attention residual: attention_ln_grad(" + 
                                   std::to_string(attention_ln_grad.rows()) + "," + 
                                   std::to_string(attention_ln_grad.cols()) + ") != residual_grad(" +
                                   std::to_string(residual_grad.rows()) + "," +
                                   std::to_string(residual_grad.cols()) + ")");
        }
        
        // Second residual addition
        Matrix final_grad = attention_ln_grad + residual_grad;
        std::cout << "Final grad dimensions after second addition: " << final_grad.rows() << "x" << final_grad.cols() << std::endl;
        
        std::cout << "=== TransformerLayer::backward END ===" << std::endl;
        return final_grad;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in TransformerLayer::backward: " << e.what() << std::endl;
        throw;
    }
}

// Transformer implementation
Transformer::Transformer(const TransformerConfig &config) : config(config) {
    std::cout << "\n=== Transformer::constructor START ===" << std::endl;
    
    // Initialize token embedding
    token_embedding = std::make_unique<TokenEmbedding>(config.vocab_size, config.hidden_size);
    
    // Initialize positional encoding
    pos_encoding = std::make_unique<PositionalEncoding>(config.max_seq_length, config.hidden_size);
    
    // Initialize transformer layers
    layers.reserve(config.num_layers);
    m_kv_caches.reserve(config.num_layers);
    for (size_t i = 0; i < config.num_layers; ++i) {
        layers.push_back(std::make_unique<TransformerLayer>(config, i));
        m_kv_caches.emplace_back(config.max_seq_length);
    }
    
    // Initialize final layer normalization
    final_ln = std::make_unique<LayerNorm>(config.hidden_size);
    
    std::cout << "=== Transformer::constructor END ===\n" << std::endl;
}

Matrix Transformer::forward(const std::vector<int> &input_tokens, bool use_cache) {
    std::cout << "\n=== Transformer::forward START ===" << std::endl;
    
    // Get embeddings
    Matrix embeddings = token_embedding->forward(input_tokens);
    
    // Add positional encodings
    Matrix position_ids(input_tokens.size(), 1);
    for (size_t i = 0; i < input_tokens.size(); ++i) {
        position_ids(i, 0) = static_cast<float>(i);
    }
    Matrix pos_encodings = pos_encoding->forward(position_ids);
    embeddings += pos_encodings;
    
    // Create causal mask for next-token prediction
    AttentionMask mask = AttentionMask::create_causal_mask(input_tokens.size());
    
    // Forward through layers
    hidden_states = embeddings;
    std::vector<Matrix> activations;
    activations.reserve(layers.size());
    
    for (size_t i = 0; i < layers.size(); ++i) {
        activations.push_back(hidden_states);
        hidden_states = layers[i]->forward(hidden_states, mask, 
            use_cache ? std::optional<KVCache>(m_kv_caches[i]) : std::nullopt);
    }
    
    // Final layer normalization
    hidden_states = final_ln->forward(hidden_states);
    
    // Store activations for backward pass
    last_hidden_states = hidden_states;
    m_layer_activations = std::move(activations);
    
    std::cout << "=== Transformer::forward END ===\n" << std::endl;
    return hidden_states;
}

void Transformer::clear_kv_cache() {
    for (auto& cache : m_kv_caches) {
        cache.clear();
    }
}

void Transformer::backward(const Matrix &grad_output, const std::vector<int> &input_tokens, float learning_rate) {
    std::cout << "\n=== Transformer::backward START ===" << std::endl;
    
    Matrix current_grad = grad_output;
    std::cout << "Initial grad dimensions: " << current_grad.rows() << "x" << current_grad.cols() << std::endl;
    
    // Backward through final layer norm
    const Matrix& last_activation = m_layer_activations.back();
    std::cout << "Last activation dimensions: " << last_activation.rows() << "x" << last_activation.cols() << std::endl;
    
    if (current_grad.rows() != last_activation.rows() || current_grad.cols() != last_activation.cols()) {
        throw std::runtime_error("Dimension mismatch in final layer norm backward: grad(" + 
                               std::to_string(current_grad.rows()) + "," + 
                               std::to_string(current_grad.cols()) + ") != activation(" +
                               std::to_string(last_activation.rows()) + "," +
                               std::to_string(last_activation.cols()) + ")");
    }
    
    current_grad = final_ln->backward(current_grad, last_activation);
    std::cout << "After final LN grad dimensions: " << current_grad.rows() << "x" << current_grad.cols() << std::endl;
    
    // Backward through layers in reverse order
    for (int i = layers.size() - 1; i >= 0; --i) {
        std::cout << "\nProcessing layer " << i << std::endl;
        const Matrix& layer_input = m_layer_activations[i];
        std::cout << "Layer input dimensions: " << layer_input.rows() << "x" << layer_input.cols() << std::endl;
        std::cout << "Current grad dimensions: " << current_grad.rows() << "x" << current_grad.cols() << std::endl;
        
        if (current_grad.rows() != layer_input.rows() || current_grad.cols() != layer_input.cols()) {
            throw std::runtime_error("Dimension mismatch in layer " + std::to_string(i) + 
                                   " backward: grad(" + std::to_string(current_grad.rows()) + 
                                   "," + std::to_string(current_grad.cols()) + 
                                   ") != input(" + std::to_string(layer_input.rows()) + 
                                   "," + std::to_string(layer_input.cols()) + ")");
        }
        
        current_grad = layers[i]->backward(current_grad, layer_input, Matrix());
        std::cout << "After layer " << i << " grad dimensions: " << current_grad.rows() << "x" << current_grad.cols() << std::endl;
    }
    
    // Update parameters
    update_parameters(learning_rate);
    
    std::cout << "=== Transformer::backward END ===\n" << std::endl;
}

void Transformer::update_parameters(float learning_rate) {
    std::cout << "=== Transformer::update_parameters START ===" << std::endl;
    
    // Update Matrix parameters
    auto& params = parameters();
    auto& grads = parameter_gradients();
    
    for (size_t i = 0; i < params.size(); ++i) {
        Matrix& param = params[i];
        const Matrix& grad = grads[i];
        
        // Update rule: param = param - learning_rate * grad
        for (size_t j = 0; j < param.size(); ++j) {
            param.data()[j] -= learning_rate * grad.data()[j];
        }
    }
    
    // Update Vector parameters for each layer
    for (auto& layer : layers) {
        // Update attention biases
        auto& attn_params = layer->self_attention->parameters();
        for (auto& bias : attn_params.vectors) {
            for (size_t j = 0; j < bias.get().size(); ++j) {
                bias.get().data()[j] -= learning_rate * 0.01f; // Small constant gradient for biases
            }
        }
        
        // Update layer norm parameters
        auto& ln_params = layer->attention_ln->parameters();
        for (auto& param : ln_params) {
            for (size_t j = 0; j < param.get().size(); ++j) {
                param.get().data()[j] -= learning_rate * 0.01f; // Small constant gradient for layer norm
            }
        }
        
        // Update feed forward biases
        auto& ffn_params = layer->feed_forward->parameters();
        for (auto& bias : ffn_params.vectors) {
            for (size_t j = 0; j < bias.get().size(); ++j) {
                bias.get().data()[j] -= learning_rate * 0.01f; // Small constant gradient for biases
            }
        }
    }
    
    // Clear gradients after update
    parameter_grads.reset();
    
    std::cout << "=== Transformer::update_parameters END ===" << std::endl;
}

std::vector<Matrix>& Transformer::parameters() {
    static std::vector<Matrix> all_params;
    all_params.clear();
    
    // Token embedding parameters (only Matrix)
    if (token_embedding) {
        auto& token_params = token_embedding->parameters();
        for (const auto& param : token_params) {
            all_params.push_back(param.get());
        }
    }
    
    // Layer parameters
    for (const auto& layer : layers) {
        // Attention parameters
        auto& attention_params = layer->self_attention->parameters();
        for (const auto& param : attention_params.matrices) {
            all_params.push_back(param.get());
        }
        
        // Layer norm parameters (only Vectors, skip)
        
        // Feed forward parameters
        auto& ffn_params = layer->feed_forward->parameters();
        for (const auto& param : ffn_params.matrices) {
            all_params.push_back(param.get());
        }
    }
    
    return all_params;
}

Transformer::~Transformer() {
    std::cout << "Transformer destructor called" << std::endl;
}