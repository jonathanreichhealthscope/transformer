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
#include <nlohmann/json.hpp>

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
    
    // Debug normalized output and validate statistics
    float min_norm = std::numeric_limits<float>::infinity();
    float max_norm = -std::numeric_limits<float>::infinity();
    float sum_norm = 0.0f;
    float sum_squared = 0.0f;
    size_t nonzero_norm = 0;
    const size_t total_elements = normalized.rows() * normalized.cols();
    
    #pragma omp parallel for collapse(2) reduction(min:min_norm) reduction(max:max_norm) \
                             reduction(+:sum_norm,sum_squared,nonzero_norm)
    for (size_t i = 0; i < normalized.rows(); i++) {
        for (size_t j = 0; j < normalized.cols(); j++) {
            float val = normalized(i, j);
            min_norm = std::min(min_norm, val);
            max_norm = std::max(max_norm, val);
            sum_norm += val;
            sum_squared += val * val;
            if (std::abs(val) > 1e-6) nonzero_norm++;
        }
    }
    
    float mean = sum_norm / total_elements;
    float variance = (sum_squared / total_elements) - (mean * mean);
    
    // Check for layer norm instability
    const float STABILITY_THRESHOLD = 1e3;
    if (std::abs(mean) > 1e-2 || std::abs(variance - 1.0) > 1e-1 || 
        std::abs(min_norm) > STABILITY_THRESHOLD || std::abs(max_norm) > STABILITY_THRESHOLD) {
        std::cerr << "WARNING: Layer normalization statistics outside expected ranges:\n"
                  << "Mean: " << mean << " (expected close to 0)\n"
                  << "Variance: " << variance << " (expected close to 1)\n"
                  << "Min: " << min_norm << "\n"
                  << "Max: " << max_norm << "\n";
                  
        // Clip extreme values if needed
        if (std::abs(min_norm) > STABILITY_THRESHOLD || std::abs(max_norm) > STABILITY_THRESHOLD) {
            for (size_t i = 0; i < normalized.rows(); i++) {
                for (size_t j = 0; j < normalized.cols(); j++) {
                    normalized(i, j) = std::max(-STABILITY_THRESHOLD, 
                                              std::min(STABILITY_THRESHOLD, normalized(i, j)));
                }
            }
            std::cerr << "Applied value clipping for stability\n";
        }
    }
    
    std::cout << "After attention layer norm:\n"
              << "Min norm: " << min_norm << "\n"
              << "Max norm: " << max_norm << "\n"
              << "Mean norm: " << mean << "\n"
              << "Variance: " << variance << "\n"
              << "Nonzero norm: " << nonzero_norm << "/" << total_elements << "\n\n";

    // Cache the normalized input for attention backward pass
    std::string attn_key = "attn_norm_" + std::to_string(layer_idx);
    GradientCheckpoint::cache_activation(attn_key, normalized);

    // Self attention
    Matrix attention_output = self_attention->forward(normalized, mask, kv_cache);
    
    // Debug attention output
    float min_attn = std::numeric_limits<float>::infinity();
    float max_attn = -std::numeric_limits<float>::infinity();
    float sum_attn = 0.0f;
    size_t nonzero_attn = 0;
    
    #pragma omp parallel for collapse(2) reduction(min:min_attn) reduction(max:max_attn) \
                             reduction(+:sum_attn,nonzero_attn)
    for (size_t i = 0; i < attention_output.rows(); i++) {
        for (size_t j = 0; j < attention_output.cols(); j++) {
            float val = attention_output(i, j);
            min_attn = std::min(min_attn, val);
            max_attn = std::max(max_attn, val);
            sum_attn += val;
            if (std::abs(val) > 1e-6) nonzero_attn++;
        }
    }
    
    std::cout << "After self attention:\n"
              << "Min attn: " << min_attn << "\n"
              << "Max attn: " << max_attn << "\n"
              << "Mean attn: " << sum_attn / (attention_output.rows() * attention_output.cols()) << "\n"
              << "Nonzero attn: " << nonzero_attn << "/" 
              << (attention_output.rows() * attention_output.cols()) << "\n\n";
    
    if (training) {
        attention_output = attention_dropout->forward(attention_output, true);
    }
    Matrix residual = attention_output + normalized;
    
    // Debug residual
    float min_res = std::numeric_limits<float>::infinity();
    float max_res = -std::numeric_limits<float>::infinity();
    float sum_res = 0.0f;
    size_t nonzero_res = 0;
    
    for (size_t i = 0; i < residual.rows(); i++) {
        for (size_t j = 0; j < residual.cols(); j++) {
            float val = residual(i, j);
            min_res = std::min(min_res, val);
            max_res = std::max(max_res, val);
            sum_res += val;
            if (std::abs(val) > 1e-6) nonzero_res++;
        }
    }
    
    std::cout << "After residual connection:\n"
              << "Min res: " << min_res << "\n"
              << "Max res: " << max_res << "\n"
              << "Mean res: " << sum_res / (residual.rows() * residual.cols()) << "\n"
              << "Nonzero res: " << nonzero_res << "/" 
              << (residual.rows() * residual.cols()) << "\n\n";
    
    std::cout << "calculating attention ln" << std::endl;
    Matrix norm1 = attention_ln->forward(residual);
    
    // Debug norm1
    float min_norm1 = std::numeric_limits<float>::infinity();
    float max_norm1 = -std::numeric_limits<float>::infinity();
    float sum_norm1 = 0.0f;
    size_t nonzero_norm1 = 0;
    
    for (size_t i = 0; i < norm1.rows(); i++) {
        for (size_t j = 0; j < norm1.cols(); j++) {
            float val = norm1(i, j);
            min_norm1 = std::min(min_norm1, val);
            max_norm1 = std::max(max_norm1, val);
            sum_norm1 += val;
            if (std::abs(val) > 1e-6) nonzero_norm1++;
        }
    }
    
    std::cout << "After second attention layer norm:\n"
              << "Min norm1: " << min_norm1 << "\n"
              << "Max norm1: " << max_norm1 << "\n"
              << "Mean norm1: " << sum_norm1 / (norm1.rows() * norm1.cols()) << "\n"
              << "Nonzero norm1: " << nonzero_norm1 << "/" 
              << (norm1.rows() * norm1.cols()) << "\n\n";

    // Cache the normalized input for feed forward backward pass
    std::string ffn_key = "ffn_norm_" + std::to_string(layer_idx);
    GradientCheckpoint::cache_activation(ffn_key, norm1);
    std::cout << "Cached normalized input for feed forward: " << norm1.rows() << "x"
                  << norm1.cols() << std::endl;
    // Feed forward
    Matrix ff_output = feed_forward->forward(norm1);
    
    // Debug feed forward output
    float min_ff = std::numeric_limits<float>::infinity();
    float max_ff = -std::numeric_limits<float>::infinity();
    float sum_ff = 0.0f;
    size_t nonzero_ff = 0;
    
    for (size_t i = 0; i < ff_output.rows(); i++) {
        for (size_t j = 0; j < ff_output.cols(); j++) {
            float val = ff_output(i, j);
            min_ff = std::min(min_ff, val);
            max_ff = std::max(max_ff, val);
            sum_ff += val;
            if (std::abs(val) > 1e-6) nonzero_ff++;
        }
    }
    
    std::cout << "After feed forward:\n"
              << "Min ff: " << min_ff << "\n"
              << "Max ff: " << max_ff << "\n"
              << "Mean ff: " << sum_ff / (ff_output.rows() * ff_output.cols()) << "\n"
              << "Nonzero ff: " << nonzero_ff << "/" 
              << (ff_output.rows() * ff_output.cols()) << "\n\n";
    
    std::cout << "FF output dimensions: " << ff_output.rows() << "x" << ff_output.cols() << std::endl;
    if (training) {
        ff_output = ffn_dropout->forward(ff_output, true);
    }
    std::cout << "FF dropout dimensions: " << ff_output.rows() << "x" << ff_output.cols() << std::endl;
    residual = ff_output + norm1;
    
    // Debug final residual
    float min_final = std::numeric_limits<float>::infinity();
    float max_final = -std::numeric_limits<float>::infinity();
    float sum_final = 0.0f;
    size_t nonzero_final = 0;
    
    for (size_t i = 0; i < residual.rows(); i++) {
        for (size_t j = 0; j < residual.cols(); j++) {
            float val = residual(i, j);
            min_final = std::min(min_final, val);
            max_final = std::max(max_final, val);
            sum_final += val;
            if (std::abs(val) > 1e-6) nonzero_final++;
        }
    }
    
    std::cout << "After final residual:\n"
              << "Min final: " << min_final << "\n"
              << "Max final: " << max_final << "\n"
              << "Mean final: " << sum_final / (residual.rows() * residual.cols()) << "\n"
              << "Nonzero final: " << nonzero_final << "/" 
              << (residual.rows() * residual.cols()) << "\n\n";
    
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
}

Matrix Transformer::forward(const std::vector<int>& input_tokens, const std::string& original_query, const Tokenizer& tokenizer, bool use_cache) {
    static const bool use_fp16 = config.use_fp16;

    // Store the input tokens and query
    last_input_tokens_ = input_tokens;
    last_input_query_ = original_query.empty() ? tokenizer.decode(input_tokens) : original_query;

    // Get embeddings and add positional encodings
    Matrix embeddings = token_embedding->forward(input_tokens);
    
    // Debug embeddings
    float min_emb = std::numeric_limits<float>::infinity();
    float max_emb = -std::numeric_limits<float>::infinity();
    float sum_emb = 0.0f;
    size_t nonzero_emb = 0;
    
    #pragma omp parallel for collapse(2) reduction(min:min_emb) reduction(max:max_emb) \
                             reduction(+:sum_emb,nonzero_emb)
    for (size_t i = 0; i < embeddings.rows(); i++) {
        for (size_t j = 0; j < embeddings.cols(); j++) {
            float val = embeddings(i, j);
            min_emb = std::min(min_emb, val);
            max_emb = std::max(max_emb, val);
            sum_emb += val;
            if (std::abs(val) > 1e-6) nonzero_emb++;
        }
    }
    
    std::cout << "\nEmbedding Statistics:\n"
              << "Min emb: " << min_emb << "\n"
              << "Max emb: " << max_emb << "\n"
              << "Mean emb: " << sum_emb / (embeddings.rows() * embeddings.cols()) << "\n"
              << "Nonzero emb: " << nonzero_emb << "/" 
              << (embeddings.rows() * embeddings.cols()) << "\n\n";
    
    // Create position_ids
    Matrix position_ids(input_tokens.size(), 1);
    #pragma omp parallel for
    for (size_t i = 0; i < input_tokens.size(); ++i) {
        position_ids(i, 0) = static_cast<float>(i);
    }
    
    // Batch process embeddings
    Matrix pos_encodings = pos_encoding->forward(position_ids);
    
    // Center positional encodings around zero
    float pos_mean = 0.0f;
    #pragma omp parallel for reduction(+:pos_mean)
    for (size_t i = 0; i < pos_encodings.rows(); i++) {
        for (size_t j = 0; j < pos_encodings.cols(); j++) {
            pos_mean += pos_encodings(i, j);
        }
    }
    pos_mean /= (pos_encodings.rows() * pos_encodings.cols());

    // Subtract mean to center around zero
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < pos_encodings.rows(); i++) {
        for (size_t j = 0; j < pos_encodings.cols(); j++) {
            pos_encodings(i, j) -= pos_mean;
        }
    }
    
    // Debug positional encodings
    float min_pos = std::numeric_limits<float>::infinity();
    float max_pos = -std::numeric_limits<float>::infinity();
    float sum_pos = 0.0f;
    size_t nonzero_pos = 0;
    
    for (size_t i = 0; i < pos_encodings.rows(); i++) {
        for (size_t j = 0; j < pos_encodings.cols(); j++) {
            float val = pos_encodings(i, j);
            min_pos = std::min(min_pos, val);
            max_pos = std::max(max_pos, val);
            sum_pos += val;
            if (std::abs(val) > 1e-6) nonzero_pos++;
        }
    }
    
    std::cout << "Positional Encoding Statistics:\n"
              << "Min pos: " << min_pos << "\n"
              << "Max pos: " << max_pos << "\n"
              << "Mean pos: " << sum_pos / (pos_encodings.rows() * pos_encodings.cols()) << "\n"
              << "Nonzero pos: " << nonzero_pos << "/" 
              << (pos_encodings.rows() * pos_encodings.cols()) << "\n\n";
    
    if (use_fp16) {
        HalfPrecisionTraining::convert_to_fp16(embeddings);
        HalfPrecisionTraining::convert_to_fp16(pos_encodings);
    }
    embeddings += pos_encodings;
    
    // Create causal mask for next-token prediction
    AttentionMask mask = AttentionMask::create_causal_mask(input_tokens.size());
    
    // Forward through layers with minimal synchronization
    hidden_states = embeddings;
    m_layer_activations.clear();
    m_layer_activations.reserve(layers.size());

    if (training && dropout) {
        hidden_states = dropout->forward(hidden_states, true);
    }

    // Normalize hidden states before processing
    float hidden_mean = 0.0f;
    float hidden_var = 0.0f;
    
    // Calculate mean
    #pragma omp parallel for reduction(+:hidden_mean)
    for (size_t i = 0; i < hidden_states.rows(); i++) {
        for (size_t j = 0; j < hidden_states.cols(); j++) {
            hidden_mean += hidden_states(i, j);
        }
    }
    hidden_mean /= (hidden_states.rows() * hidden_states.cols());
    
    // Calculate variance
    #pragma omp parallel for reduction(+:hidden_var)
    for (size_t i = 0; i < hidden_states.rows(); i++) {
        for (size_t j = 0; j < hidden_states.cols(); j++) {
            float diff = hidden_states(i, j) - hidden_mean;
            hidden_var += diff * diff;
        }
    }
    hidden_var /= (hidden_states.rows() * hidden_states.cols());
    
    // Normalize with scaling factor to keep values in reasonable range
    const float eps = 1e-5f;
    const float scale = std::sqrt(2.0f);  // Scale to get variance closer to 1
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < hidden_states.rows(); i++) {
        for (size_t j = 0; j < hidden_states.cols(); j++) {
            hidden_states(i, j) = scale * (hidden_states(i, j) - hidden_mean) / std::sqrt(hidden_var + eps);
        }
    }

    // Add gradient clipping for stability
    const float clip_threshold = 1.0f;
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < hidden_states.rows(); i++) {
        for (size_t j = 0; j < hidden_states.cols(); j++) {
            hidden_states(i, j) = std::max(-clip_threshold, 
                                         std::min(clip_threshold, hidden_states(i, j)));
        }
    }

    // Process layers with minimal synchronization
    for (size_t i = 0; i < layers.size(); ++i) {
        try {
            m_layer_activations.push_back(hidden_states);
            
            // Debug hidden states before layer
            float min_hidden = std::numeric_limits<float>::infinity();
            float max_hidden = -std::numeric_limits<float>::infinity();
            float sum_hidden = 0.0f;
            size_t nonzero_hidden = 0;
            
            #pragma omp parallel for collapse(2) reduction(min:min_hidden) reduction(max:max_hidden) \
                                 reduction(+:sum_hidden,nonzero_hidden)
            for (size_t r = 0; r < hidden_states.rows(); r++) {
                for (size_t c = 0; c < hidden_states.cols(); c++) {
                    float val = hidden_states(r, c);
                    min_hidden = std::min(min_hidden, val);
                    max_hidden = std::max(max_hidden, val);
                    sum_hidden += val;
                    if (std::abs(val) > 1e-6) nonzero_hidden++;
                }
            }
            
            // Apply residual scaling to prevent value explosion
            const float residual_scale = 0.7071f; // 1/√2
            #pragma omp parallel for collapse(2)
            for (size_t r = 0; r < hidden_states.rows(); r++) {
                for (size_t c = 0; c < hidden_states.cols(); c++) {
                    hidden_states(r, c) *= residual_scale;
                }
            }

            hidden_states = layers[i]->forward(hidden_states, mask,
                                             use_cache ? std::optional<KVCache>(m_kv_caches[i])
                                                     : std::nullopt);
        } catch (const std::exception& e) {
            std::cerr << "Error in layer " << i << ": " << e.what() << std::endl;
            throw;
        }
    }

    // Final normalization
    hidden_states = final_ln->forward(hidden_states);
    
    // Debug final hidden states
    float min_final = std::numeric_limits<float>::infinity();
    float max_final = -std::numeric_limits<float>::infinity();
    float sum_final = 0.0f;
    size_t nonzero_final = 0;
    
    for (size_t i = 0; i < hidden_states.rows(); i++) {
        for (size_t j = 0; j < hidden_states.cols(); j++) {
            float val = hidden_states(i, j);
            min_final = std::min(min_final, val);
            max_final = std::max(max_final, val);
            sum_final += val;
            if (std::abs(val) > 1e-6) nonzero_final++;
        }
    }
    
    std::cout << "Final Hidden States Statistics:\n"
              << "Min final: " << min_final << "\n"
              << "Max final: " << max_final << "\n"
              << "Mean final: " << sum_final / (hidden_states.rows() * hidden_states.cols()) << "\n"
              << "Nonzero final: " << nonzero_final << "/" 
              << (hidden_states.rows() * hidden_states.cols()) << "\n\n";
    
    // Single sync point at the end
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    
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
                           const Matrix& target_distribution,
                           const Tokenizer& tokenizer) {
    // Forward pass
    std::vector<Matrix> batch_outputs;
    for (const auto& tokens : input_tokens) {
        batch_outputs.push_back(forward(tokens, "", tokenizer));
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

    #pragma omp parallel for
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

        #pragma omp parallel for collapse(2)
        for (size_t j = 0; j < param.rows(); ++j) {
            for (size_t k = 0; k < param.cols(); ++k) {
                param(j, k) -= learning_rate * grad(j, k);
            }
        }
    }

    // Update Vector parameters for each layer
    #pragma omp parallel for
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        std::cout << "\nProcessing layer " << layer_idx << std::endl;
        auto& layer = layers[layer_idx];

        // Update attention parameters
        auto& attn_params = layer->self_attention->parameters();
        auto& attn_grads = layer->self_attention->parameter_gradients();

        std::cout << "Attention vectors: " << attn_params.vectors.size() << " parameters, "
                  << attn_grads.vectors.size() << " gradients" << std::endl;

        // Update attention biases using computed gradients
        #pragma omp parallel for
        for (size_t i = 0; i < attn_params.vectors.size(); ++i) {
            auto& bias = attn_params.vectors[i];
            const auto& bias_grad = attn_grads.vectors[i];

            std::cout << "Attention bias " << i << ": bias size=" << bias.get().size()
                      << ", grad size=" << bias_grad.get().size() << std::endl;

            if (bias.get().size() != bias_grad.get().size()) {
                throw std::runtime_error("Dimension mismatch in attention bias update");
            }

            #pragma omp parallel for
            for (size_t j = 0; j < bias.get().size(); ++j) {
                bias.get().data()[j] -= learning_rate * bias_grad.get().data()[j];
            }
        }

        // Update layer norm parameters
        auto& ln_params = layer->attention_ln->parameters();
        auto& ln_grads = layer->attention_ln->parameter_gradients();

        std::cout << "Layer norm vectors: " << ln_params.size() << " parameters, "
                  << ln_grads.size() << " gradients" << std::endl;

        #pragma omp parallel for
        for (size_t i = 0; i < ln_params.size(); ++i) {
            auto& param = ln_params[i];
            const auto& grad = ln_grads[i];

            std::cout << "Layer norm param " << i << ": param size=" << param.get().size()
                      << ", grad size=" << grad.get().size() << std::endl;

            if (param.get().size() != grad.get().size()) {
                throw std::runtime_error("Dimension mismatch in layer norm update");
            }

            #pragma omp parallel for
            for (size_t j = 0; j < param.get().size(); ++j) {
                param.get().data()[j] -= learning_rate * grad.get().data()[j];
            }
        }

        // Update feed forward parameters
        auto& ffn_params = layer->feed_forward->parameters();
        auto& ffn_grads = layer->feed_forward->parameter_gradients();

        std::cout << "Feed forward vectors: " << ffn_params.vectors.size() << " parameters, "
                  << ffn_grads.vectors.size() << " gradients" << std::endl;

        #pragma omp parallel for
        for (size_t i = 0; i < ffn_params.vectors.size(); ++i) {
            auto& bias = ffn_params.vectors[i];
            const auto& bias_grad = ffn_grads.vectors[i];

            std::cout << "Feed forward bias " << i << ": bias size=" << bias.get().size()
                      << ", grad size=" << bias_grad.get().size() << std::endl;

            if (bias.get().size() != bias_grad.get().size()) {
                throw std::runtime_error("Dimension mismatch in feed forward bias update");
            }

            #pragma omp parallel for
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

std::vector<Matrix>& Transformer::parameter_gradients() {
    if (!parameter_grads) {
        // Initialize parameter gradients if they don't exist
        parameter_grads = std::vector<Matrix>();
        auto& params = parameters();
        parameter_grads->reserve(params.size());
        
        // Create gradient matrices with same dimensions as parameters
        for (const auto& param : params) {
            parameter_grads->emplace_back(param.rows(), param.cols());
        }
    }
    return *parameter_grads;
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

void TransformerConfig::load_from_json(const std::string& config_path) {
    try {
        // Read and parse JSON file
        std::ifstream file(config_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open config file: " + config_path);
        }
        
        nlohmann::json json_config;
        file >> json_config;
        
        // Load model parameters
        if (json_config.contains("model")) {
            const auto& model = json_config["model"];
            if (model.contains("vocab_size")) {
                vocab_size = model["vocab_size"];
            }
            if (model.contains("hidden_size")) {
                hidden_size = model["hidden_size"];
            }
            if (model.contains("num_heads")) {
                num_heads = model["num_heads"];
            }
            if (model.contains("num_layers")) {
                num_layers = model["num_layers"];
            }
            if (model.contains("head_dim")) {
                head_dim = model["head_dim"];
            }
            if (model.contains("intermediate_size")) {
                intermediate_size = model["intermediate_size"];
            }
            if (model.contains("max_seq_length")) {
                max_seq_length = model["max_seq_length"];
            }
        }

        // Load attention parameters
        if (json_config.contains("attention")) {
            const auto& attention = json_config["attention"];
            if (attention.contains("use_flash_attention")) {
                use_flash_attention = attention["use_flash_attention"];
            }
            if (attention.contains("use_rope")) {
                use_rope = attention["use_rope"];
            }
            if (attention.contains("use_sliding_window")) {
                use_sliding_window = attention["use_sliding_window"];
            }
            if (attention.contains("window_size")) {
                window_size = attention["window_size"];
            }
            if (attention.contains("use_gqa")) {
                use_gqa = attention["use_gqa"];
            }
            if (attention.contains("num_kv_heads")) {
                num_kv_heads = attention["num_kv_heads"];
            }
        }

        // Load optimization parameters
        if (json_config.contains("optimization")) {
            const auto& optimization = json_config["optimization"];
            if (optimization.contains("use_fp16")) {
                use_fp16 = optimization["use_fp16"];
            }
        }

        // Load tokenizer configuration
        if (json_config.contains("tokenizer")) {
            const auto& tok_config = json_config["tokenizer"];
            if (tok_config.contains("vocab_size")) {
                tokenizer.vocab_size = tok_config["vocab_size"];
            }
            if (tok_config.contains("model_path")) {
                tokenizer.model_path = tok_config["model_path"];
            }
            if (tok_config.contains("use_subword")) {
                tokenizer.use_subword = tok_config["use_subword"];
            }
            if (tok_config.contains("special_tokens")) {
                tokenizer.special_tokens = tok_config["special_tokens"].get<std::vector<std::string>>();
            }
        }

    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading config from JSON: " + std::string(e.what()));
    }
}

void Transformer::set_training(bool training_mode) {
    training = training_mode;
    // Set training mode for all components that need it
    for (auto& layer : layers) {
        layer->training = training_mode;
    }
    if (lm_head) {
        lm_head->set_training(training_mode);
    }
}

std::pair<std::string, PhraseType> Transformer::predict_final_phrase(
    const std::string& input_text,
    const Tokenizer& tokenizer
) {
    // First predict the phrase type
    PhraseType predicted_type = predict_phrase_type(input_text, tokenizer);
    
    // Tokenize input without delimiter
    std::vector<int> tokens = tokenizer.encode(input_text);
    
    // Forward pass
    Matrix logits = forward(tokens, input_text, tokenizer);
    
    // Extract the prediction based on the predicted type
    std::string predicted_phrase = extract_prediction(logits, predicted_type, tokenizer);
    
    return {predicted_phrase, predicted_type};
}

PhraseType Transformer::predict_phrase_type(
    const std::string& input_text,
    const Tokenizer& tokenizer
) {
    // Tokenize input
    std::vector<int> tokens = tokenizer.encode(input_text);
    
    // Forward pass
    Matrix logits = forward(tokens, input_text, tokenizer);
    
    // Analyze logits to determine phrase type
    return analyze_phrase_type(logits, tokenizer);
}

PhraseType Transformer::analyze_phrase_type(
    const Matrix& logits,
    const Tokenizer& tokenizer
) {
    // Get the final token predictions
    Matrix final_logits = Matrix(logits.row(logits.rows() - 1));  // Explicit conversion
    
    // Calculate scores for each phrase type based on token probabilities
    float verb_score = 0.0f;
    float adj_score = 0.0f;
    float general_score = 0.0f;
    
    // Analyze token probabilities to determine the most likely phrase type
    // This would involve checking against known verb and adjective patterns
    // in the vocabulary
    
    // For now, using a simple heuristic based on highest probability tokens
    // This should be replaced with a more sophisticated analysis based on
    // your specific vocabulary and token patterns
    
    if (verb_score > adj_score && verb_score > general_score) {
        return PhraseType::VERB;
    } else if (adj_score > verb_score && adj_score > general_score) {
        return PhraseType::ADJECTIVE;
    }
    
    return PhraseType::GENERAL;
}

std::string Transformer::extract_prediction(
    const Matrix& logits,
    PhraseType phrase_type,
    const Tokenizer& tokenizer
) {
    // Get the final token predictions
    Matrix final_logits = Matrix(logits.row(logits.rows() - 1));  // Explicit conversion
    
    // Apply softmax with temperature
    const float temperature = 0.7f;  // Lower = more focused predictions, higher = more diverse
    float max_logit = -std::numeric_limits<float>::infinity();
    
    // Find max for numerical stability
    for (size_t i = 0; i < final_logits.cols(); i++) {
        max_logit = std::max(max_logit, final_logits(0, i));
    }
    
    // Compute softmax probabilities
    std::vector<float> probabilities(final_logits.cols());
    float sum_exp = 0.0f;
    
    for (size_t i = 0; i < final_logits.cols(); i++) {
        float scaled_logit = (final_logits(0, i) - max_logit) / temperature;
        probabilities[i] = std::exp(scaled_logit);
        sum_exp += probabilities[i];
    }
    
    // Normalize probabilities
    for (float& prob : probabilities) {
        prob /= sum_exp;
    }
    
    // Apply type-specific boosts
    switch (phrase_type) {
        case PhraseType::VERB:
            boost_verb_probabilities(probabilities, tokenizer);
            break;
        case PhraseType::ADJECTIVE:
            boost_adjective_probabilities(probabilities, tokenizer);
            break;
        default:
            break;
    }
    
    // Sample from the distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
    int predicted_token = dist(gen);
    
    // Decode the predicted token
    return tokenizer.decode({predicted_token});
}

void Transformer::boost_verb_probabilities(std::vector<float>& probabilities, const Tokenizer& tokenizer) {
    const float boost_factor = 1.5f;
    for (size_t i = 0; i < probabilities.size(); i++) {
        std::string token = tokenizer.decode({static_cast<int>(i)});
        if (is_likely_verb(token)) {
            probabilities[i] *= boost_factor;
        }
    }
    // Renormalize
    float sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0f);
    for (float& prob : probabilities) {
        prob /= sum;
    }
}

void Transformer::boost_adjective_probabilities(std::vector<float>& probabilities, const Tokenizer& tokenizer) {
    const float boost_factor = 1.5f;
    for (size_t i = 0; i < probabilities.size(); i++) {
        std::string token = tokenizer.decode({static_cast<int>(i)});
        if (is_likely_adjective(token)) {
            probabilities[i] *= boost_factor;
        }
    }
    // Renormalize
    float sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0f);
    for (float& prob : probabilities) {
        prob /= sum;
    }
}

bool Transformer::is_likely_verb(const std::string& token) {
    const std::vector<std::string> verb_endings = {"ing", "ed", "ate", "ize", "ify"};
    for (const auto& ending : verb_endings) {
        if (token.length() > ending.length() && 
            token.substr(token.length() - ending.length()) == ending) {
            return true;
        }
    }
    return false;
}

bool Transformer::is_likely_adjective(const std::string& token) {
    const std::vector<std::string> adj_endings = {"ful", "ous", "ible", "able", "al", "ive"};
    for (const auto& ending : adj_endings) {
        if (token.length() > ending.length() && 
            token.substr(token.length() - ending.length()) == ending) {
            return true;
        }
    }
    return false;
}