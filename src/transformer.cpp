#include "../include/transformer.hpp"
#include "../include/cuda/cublas_check.cuh"
#include "../include/cuda/cuda_check.cuh"
#include "../include/logger.hpp"
#include "../include/half_precision.hpp"
#include <cublas_v2.h>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <stdexcept>

extern cublasHandle_t cublas_handle;

// TransformerLayer implementation
TransformerLayer::TransformerLayer(const TransformerConfig& config_, size_t idx)
    : config(config_), layer_idx(idx) {
    // Initialize components
    std::cout << "Initializing TransformerLayer " << idx << " with GQA config:" << std::endl;
    std::cout << "- use_gqa: " << (config.use_gqa ? "true" : "false") << std::endl;
    std::cout << "- num_kv_heads: " << config.num_kv_heads << std::endl;
    std::cout << "- hidden_size: " << config.hidden_size << std::endl;
    std::cout << "- intermediate_size: " << config.intermediate_size << std::endl;

    self_attention = std::make_unique<MultiHeadAttention>(
        config.hidden_size, config.num_heads, config.head_dim, config.dropout_prob,
        config.use_flash_attention, config.use_rope, config.use_sliding_window, config.window_size,
        config.use_gqa, config.num_kv_heads, config.max_seq_length, config.use_fp16);

    attention_ln = std::make_unique<LayerNorm>(config.hidden_size);
    feed_forward = std::make_unique<FeedForward>(config.hidden_size, config.intermediate_size);
    ffn_ln = std::make_unique<LayerNorm>(config.hidden_size);

    // Initialize dropout layers
    attention_dropout = std::make_unique<Dropout>(config.dropout_rate);
    ffn_dropout = std::make_unique<Dropout>(config.dropout_rate);
}

Matrix TransformerLayer::forward(const Matrix& input, const AttentionMask& mask,
                                 const std::optional<KVCache>& kv_cache) {
    std::cout << "=== TransformerLayer::forward START ===" << std::endl;

    // Layer norm before attention
    Matrix normalized = attention_ln->forward(input);

    // Cache the normalized input for attention backward pass
    std::string attn_key = "attn_norm_" + std::to_string(layer_idx);
    GradientCheckpoint::cache_activation(attn_key, normalized);

    // Self attention
    Matrix attention_output = self_attention->forward(normalized, mask, kv_cache);
    if (training) {
        attention_output = attention_dropout->forward(attention_output, true);
    }
    Matrix residual = attention_output + normalized;
    std::cout << "calculating attention ln" << std::endl;
    Matrix norm1 = attention_ln->forward(residual);

    // Cache the normalized input for feed forward backward pass
    std::string ffn_key = "ffn_norm_" + std::to_string(layer_idx);
    GradientCheckpoint::cache_activation(ffn_key, norm1);
    std::cout << "Cached normalized input for feed forward: " << norm1.rows() << "x"
                  << norm1.cols() << std::endl;
    // Feed forward
    Matrix ff_output = feed_forward->forward(norm1);
    std::cout << "FF output dimensions: " << ff_output.rows() << "x" << ff_output.cols() << std::endl;
    if (training) {
        ff_output = ffn_dropout->forward(ff_output, true);
    }
    std::cout << "FF dropout dimensions: " << ff_output.rows() << "x" << ff_output.cols() << std::endl;
    residual = ff_output + norm1;
    std::cout << "Residual dimensions: " << residual.rows() << "x" << residual.cols() << std::endl;
    return ffn_ln->forward(residual);
}

Matrix TransformerLayer::backward(const Matrix& grad_output, const Matrix& input,
                                  const Matrix& target_distribution) {
    std::cout << "=== TransformerLayer::backward START ===" << std::endl;
    std::cout << "Grad output dimensions: " << grad_output.rows() << "x" << grad_output.cols()
              << std::endl;
    std::cout << "Input dimensions: " << input.rows() << "x" << input.cols() << std::endl;

    try {
        // Get the cached normalized input for feed forward
        std::cout << "Getting cached normalized input for feed forward" << std::endl;
        std::string ffn_key = "ffn_norm_" + std::to_string(layer_idx);
        Matrix ffn_normalized = GradientCheckpoint::get_activation(ffn_key);
        std::cout << "Cached normalized input for feed forward: " << ffn_normalized.rows() << "x"
                  << ffn_normalized.cols() << std::endl;

        // Backward through feed forward network
        Matrix ff_dropout_grad = training ? ffn_dropout->backward(grad_output) : grad_output;
        std::cout << "FF dropout grad dimensions: " << ff_dropout_grad.rows() << "x"
                  << ff_dropout_grad.cols() << std::endl;
        Matrix ffn_grad = feed_forward->backward(ff_dropout_grad, ffn_normalized);
        std::cout << "FFN grad dimensions: " << ffn_grad.rows() << "x" << ffn_grad.cols()
                  << std::endl;

        // Backward through feed forward layer norm
        Matrix ffn_ln_grad = ffn_ln->backward(ffn_grad, input);
        std::cout << "FFN LN grad dimensions: " << ffn_ln_grad.rows() << "x" << ffn_ln_grad.cols()
                  << std::endl;

        // Check dimensions before first residual addition
        std::cout << "About to add FFN LN grad (" << ffn_ln_grad.rows() << "x" << ffn_ln_grad.cols()
                  << ") with grad_output (" << grad_output.rows() << "x" << grad_output.cols()
                  << ")" << std::endl;

        if (ffn_ln_grad.rows() != grad_output.rows() || ffn_ln_grad.cols() != grad_output.cols()) {
            throw std::runtime_error("Dimension mismatch in FFN residual: ffn_ln_grad(" +
                                     std::to_string(ffn_ln_grad.rows()) + "," +
                                     std::to_string(ffn_ln_grad.cols()) + ") != grad_output(" +
                                     std::to_string(grad_output.rows()) + "," +
                                     std::to_string(grad_output.cols()) + ")");
        }

        // First residual addition
        Matrix residual_grad = ffn_ln_grad + grad_output;
        std::cout << "Residual grad dimensions after first addition: " << residual_grad.rows()
                  << "x" << residual_grad.cols() << std::endl;

        // Get the cached normalized input for attention
        std::string attn_key = "attn_norm_" + std::to_string(layer_idx);
        Matrix attn_normalized = GradientCheckpoint::get_activation(attn_key);
        std::cout << "Cached normalized input for attention: " << attn_normalized.rows() << "x"
                  << attn_normalized.cols() << std::endl;
        // Backward through self attention
        Matrix attn_dropout_grad =
            training ? attention_dropout->backward(residual_grad) : residual_grad;
        std::cout << "Attention dropout grad dimensions: " << attn_dropout_grad.rows() << "x"
                  << attn_dropout_grad.cols() << std::endl;
        Matrix attention_grad =
            self_attention->backward(attn_dropout_grad, attn_normalized, target_distribution);
        std::cout << "Attention grad dimensions: " << attention_grad.rows() << "x"
                  << attention_grad.cols() << std::endl;

        // Backward through attention layer norm
        Matrix attention_ln_grad = attention_ln->backward(attention_grad, input);
        std::cout << "Attention LN grad dimensions: " << attention_ln_grad.rows() << "x"
                  << attention_ln_grad.cols() << std::endl;

        // Check dimensions before second residual addition
        std::cout << "About to add attention LN grad (" << attention_ln_grad.rows() << "x"
                  << attention_ln_grad.cols() << ") with residual_grad (" << residual_grad.rows()
                  << "x" << residual_grad.cols() << ")" << std::endl;

        if (attention_ln_grad.rows() != residual_grad.rows() ||
            attention_ln_grad.cols() != residual_grad.cols()) {
            throw std::runtime_error(
                "Dimension mismatch in attention residual: attention_ln_grad(" +
                std::to_string(attention_ln_grad.rows()) + "," +
                std::to_string(attention_ln_grad.cols()) + ") != residual_grad(" +
                std::to_string(residual_grad.rows()) + "," + std::to_string(residual_grad.cols()) +
                ")");
        }

        // Second residual addition
        Matrix final_grad = attention_ln_grad + residual_grad;
        std::cout << "Final grad dimensions after second addition: " << final_grad.rows() << "x"
                  << final_grad.cols() << std::endl;

        std::cout << "=== TransformerLayer::backward END ===" << std::endl;
        return final_grad;

    } catch (const std::exception& e) {
        std::cerr << "Error in TransformerLayer::backward: " << e.what() << std::endl;
        throw;
    }
}

// Transformer implementation
Transformer::Transformer(const TransformerConfig& config) : config(config) {
    std::cout << "\n=== Transformer::constructor START ===" << std::endl;

    // Initialize dropout with config probability
    dropout = std::make_unique<Dropout>(config.dropout_prob);

    // Xavier/Glorot initialization with bounds
    auto init_weight = [](float fan_in, float fan_out) -> float {
        float limit = std::sqrt(6.0f / (fan_in + fan_out));
        limit = std::min(limit, 0.1f); // Cap maximum initialization value
        return (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * limit;
    };

    // Initialize token embedding with bounded values
    token_embedding = std::make_unique<TokenEmbedding>(config.vocab_size, config.hidden_size);

    // Initialize positional encoding
    pos_encoding = std::make_unique<PositionalEncoding>(config.max_seq_length, config.hidden_size);

    // Initialize transformer layers with bounded initialization
    layers.reserve(config.num_layers);
    m_kv_caches.reserve(config.num_layers);
    for (size_t i = 0; i < config.num_layers; ++i) {
        layers.push_back(std::make_unique<TransformerLayer>(config, i));
        m_kv_caches.emplace_back(config.max_seq_length);
    }

    // Initialize final layer normalization
    final_ln = std::make_unique<LayerNorm>(config.hidden_size);

    // Initialize the language model head with bounded values
    lm_head = std::make_unique<LanguageModelHead>(config.hidden_size, config.vocab_size);

    std::cout << "=== Transformer::constructor END ===\n" << std::endl;
}

Matrix Transformer::forward(const std::vector<int>& input_tokens, bool use_cache) {
    static const bool use_fp16 = config.use_fp16;

    // Get embeddings
    Matrix embeddings = token_embedding->forward(input_tokens);
    std::cout << "embedding dimensions: " << embeddings.shape() << std::endl;
    if (use_fp16) {
        HalfPrecisionTraining::convert_to_fp16(embeddings);
    }

    // Add positional encodings
    Matrix position_ids(input_tokens.size(), 1);
    std::cout << "position_ids dimensions: " << position_ids.shape() << std::endl;
    for (size_t i = 0; i < input_tokens.size(); ++i) {
        position_ids(i, 0) = static_cast<float>(i);
    }
    Matrix pos_encodings = pos_encoding->forward(position_ids);
    std::cout << "pos_encodings dimensions: " << pos_encodings.shape() << std::endl;
    if (use_fp16) {
        HalfPrecisionTraining::convert_to_fp16(pos_encodings);
    }

    embeddings += pos_encodings;
    std::cout << "embeddings + pos_encodings dimensions: " << embeddings.shape() << std::endl;
    // Create causal mask for next-token prediction
    AttentionMask mask = AttentionMask::create_causal_mask(input_tokens.size());
    
    // Forward through layers with stability checks
    hidden_states = embeddings;
    m_layer_activations.clear();                // Clear previous activations
    m_layer_activations.reserve(layers.size()); // Reserve space for efficiency

    // Add dropout after embeddings
    if (training && dropout) {
        hidden_states = dropout->forward(hidden_states, true);
        std::cout << "hidden_states dimensions after dropout: " << hidden_states.shape() << std::endl;
    }

    for (size_t i = 0; i < layers.size(); ++i) {
        try {
            m_layer_activations.push_back(hidden_states);
            hidden_states = layers[i]->forward(hidden_states, mask,
                                               use_cache ? std::optional<KVCache>(m_kv_caches[i])
                                                         : std::nullopt);
            std::cout << "hidden_states dimensions after layer " << i << ": " << hidden_states.shape() << std::endl;
            // Convert back to FP32 at the end
            if (use_fp16 && i == layers.size() - 1) {
                HalfPrecisionTraining::convert_to_fp32(hidden_states);
            }

            // Add dropout between layers
            if (training && dropout && i < layers.size() - 1) {
                hidden_states = dropout->forward(hidden_states, true);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in layer " << i << ": " << e.what() << std::endl;
            throw;
        }
    }

    // Store final hidden states for backward pass
    last_hidden_states = hidden_states;

    return hidden_states;
}

void Transformer::clear_kv_cache() {
    for (auto& cache : m_kv_caches) {
        cache.clear();
    }
}

// Original backward method implementation
void Transformer::backward(const Matrix& grad_output, const std::vector<int>& input_tokens, 
                         float learning_rate) {
    static const bool use_fp16 = config.use_fp16;
    
    Matrix grad = grad_output;
    if (use_fp16) {
        HalfPrecisionTraining::convert_to_fp16(grad);
    }
    // ... rest of original implementation
}

// New batch backward method implementation
void Transformer::backward(std::vector<Matrix>& outputs, const Matrix& target_distribution,
                         float learning_rate) {
    static const bool use_fp16 = config.use_fp16;
    
    Matrix grad = outputs.back();
    if (use_fp16) {
        HalfPrecisionTraining::convert_to_fp16(grad);
    }
    // ... rest of batch implementation
}

void Transformer::train_step(const std::vector<std::vector<int>>& input_tokens,
                           const Matrix& target_distribution) {
    // Forward pass
    std::vector<Matrix> batch_outputs;
    for (const auto& tokens : input_tokens) {
        batch_outputs.push_back(forward(tokens));
    }
    
    // Debug weight updates
    auto params_before = parameters();
    
    // Backward pass with learning rate
    const float learning_rate = 1e-4;  // You might want to make this configurable
    backward(batch_outputs, target_distribution, learning_rate);
    
    // Check if weights are changing
    auto params_after = parameters();
    float max_weight_change = 0.0f;
    for (size_t i = 0; i < params_before.size(); ++i) {
        Matrix diff = params_after[i] - params_before[i];
        for (size_t j = 0; j < diff.size(); ++j) {
            max_weight_change = std::max(max_weight_change, std::abs(diff.data()[j]));
        }
    }
    std::cout << "Max weight change in training step: " << max_weight_change << std::endl;
}

void Transformer::update_parameters(float learning_rate) {
    std::cout << "=== Transformer::update_parameters START ===" << std::endl;

    // Update Matrix parameters
    auto& params = parameters();
    auto& grads = parameter_gradients();

    std::cout << "Number of matrix parameters: " << params.size() << std::endl;
    std::cout << "Number of matrix gradients: " << grads.size() << std::endl;

    for (size_t i = 0; i < params.size(); ++i) {
        Matrix& param = params[i];
        const Matrix& grad = grads[i];

        std::cout << "Updating matrix parameter " << i << ": ";
        std::cout << "param shape=" << param.rows() << "x" << param.cols()
                  << ", grad shape=" << grad.rows() << "x" << grad.cols() << std::endl;

        if (param.rows() != grad.rows() || param.cols() != grad.cols()) {
            throw std::runtime_error(
                "Dimension mismatch in matrix update: param(" + std::to_string(param.rows()) + "," +
                std::to_string(param.cols()) + ") != grad(" + std::to_string(grad.rows()) + "," +
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
        auto& layer = layers[layer_idx];

        // Update attention parameters
        auto& attn_params = layer->self_attention->parameters();
        auto& attn_grads = layer->self_attention->parameter_gradients();

        std::cout << "Attention vectors: " << attn_params.vectors.size() << " parameters, "
                  << attn_grads.vectors.size() << " gradients" << std::endl;

        // Update attention biases using computed gradients
        for (size_t i = 0; i < attn_params.vectors.size(); ++i) {
            auto& bias = attn_params.vectors[i];
            const auto& bias_grad = attn_grads.vectors[i];

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
        auto& ln_params = layer->attention_ln->parameters();
        auto& ln_grads = layer->attention_ln->parameter_gradients();

        std::cout << "Layer norm vectors: " << ln_params.size() << " parameters, "
                  << ln_grads.size() << " gradients" << std::endl;

        for (size_t i = 0; i < ln_params.size(); ++i) {
            auto& param = ln_params[i];
            const auto& grad = ln_grads[i];

            std::cout << "Layer norm param " << i << ": param size=" << param.get().size()
                      << ", grad size=" << grad.get().size() << std::endl;

            if (param.get().size() != grad.get().size()) {
                throw std::runtime_error("Dimension mismatch in layer norm update");
            }

            for (size_t j = 0; j < param.get().size(); ++j) {
                param.get().data()[j] -= learning_rate * grad.get().data()[j];
            }
        }

        // Update feed forward parameters
        auto& ffn_params = layer->feed_forward->parameters();
        auto& ffn_grads = layer->feed_forward->parameter_gradients();

        std::cout << "Feed forward vectors: " << ffn_params.vectors.size() << " parameters, "
                  << ffn_grads.vectors.size() << " gradients" << std::endl;

        for (size_t i = 0; i < ffn_params.vectors.size(); ++i) {
            auto& bias = ffn_params.vectors[i];
            const auto& bias_grad = ffn_grads.vectors[i];

            std::cout << "Feed forward bias " << i << ": bias size=" << bias.get().size()
                      << ", grad size=" << bias_grad.get().size() << std::endl;

            if (bias.get().size() != bias_grad.get().size()) {
                throw std::runtime_error("Dimension mismatch in feed forward bias update");
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

void Transformer::load(std::istream& is) {
    try {
        // Load token embedding
        token_embedding->load(is);

        // Load positional encoding
        pos_encoding->load(is);

        // Load transformer layers
        for (auto& layer : layers) {
            layer->load(is);
        }

        // Load final layer norm
        final_ln->load(is);

        // Load language model head
        lm_head->load(is);

    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading transformer: " + std::string(e.what()));
    }
}