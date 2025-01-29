#include "../include/transformer.hpp"
#include "../include/cuda/cublas_check.cuh"
#include "../include/cuda/cuda_check.cuh"
#include "../include/logger.hpp"
#include "../include/half_precision.hpp"
#include "../include/training/dynamic_loss_scaler.hpp"
#include <cublas_v2.h>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <nlohmann/json.hpp>

extern cublasHandle_t cublas_handle;

// LearningRateScheduler implementation
class LearningRateScheduler {
public:
    LearningRateScheduler(float initial_lr = 0.0001f, float peak_lr = 0.001f, 
                         int warmup_steps = 1000, float decay_factor = 0.9f)
        : initial_lr_(initial_lr), peak_lr_(peak_lr), 
          warmup_steps_(warmup_steps), decay_factor_(decay_factor) {}

    float get_lr(int step) {
        if (step < warmup_steps_) {
            // Linear warmup
            return initial_lr_ + (peak_lr_ - initial_lr_) * (static_cast<float>(step) / warmup_steps_);
        }
        // Exponential decay after warmup
        return peak_lr_ * std::pow(decay_factor_, (step - warmup_steps_) / 1000.0f);
    }

private:
    float initial_lr_;
    float peak_lr_;
    int warmup_steps_;
    float decay_factor_;
};

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
    std::cout << "Grad output dimensions: " << grad_output.rows() << "x" << grad_output.cols() << std::endl;
    std::cout << "Input dimensions: " << input.rows() << "x" << input.cols() << std::endl;

    try {
        // Ensure dimensions match input
        if (grad_output.cols() != input.cols()) {
            throw std::runtime_error("Gradient output columns (" + std::to_string(grad_output.cols()) + 
                                   ") must match input columns (" + std::to_string(input.cols()) + ")");
        }

        // Get the cached normalized input for feed forward
        std::string ffn_key = "ffn_norm_" + std::to_string(layer_idx);
        Matrix ffn_normalized = GradientCheckpoint::get_activation(ffn_key);
        std::cout << "Cached normalized input for feed forward: " << ffn_normalized.rows() << "x"
                  << ffn_normalized.cols() << std::endl;

        // Backward through feed forward network with dimension validation
        Matrix ff_dropout_grad = training ? ffn_dropout->backward(grad_output) : grad_output;
        if (ff_dropout_grad.cols() != input.cols()) {
            throw std::runtime_error("FFN dropout gradient columns must match input columns");
        }

        Matrix ffn_grad = feed_forward->backward(ff_dropout_grad, ffn_normalized);
        if (ffn_grad.cols() != input.cols()) {
            throw std::runtime_error("FFN gradient columns must match input columns");
        }

        // Backward through feed forward layer norm
        Matrix ffn_ln_grad = ffn_ln->backward(ffn_grad, input);
        if (ffn_ln_grad.cols() != input.cols()) {
            throw std::runtime_error("FFN layer norm gradient columns must match input columns");
        }

        // First residual connection - proper gradient flow
        Matrix residual_grad = ffn_ln_grad + grad_output;  // Add both gradient paths
        std::cout << "Residual grad dimensions after first connection: " << residual_grad.rows()
                  << "x" << residual_grad.cols() << std::endl;

        // Get the cached normalized input for attention
        std::string attn_key = "attn_norm_" + std::to_string(layer_idx);
        Matrix attn_normalized = GradientCheckpoint::get_activation(attn_key);
        std::cout << "Cached normalized input for attention: " << attn_normalized.rows() << "x"
                  << attn_normalized.cols() << std::endl;

        // Backward through self attention with dimension validation
        Matrix attn_dropout_grad = training ? attention_dropout->backward(residual_grad) : residual_grad;
        if (attn_dropout_grad.cols() != input.cols()) {
            throw std::runtime_error("Attention dropout gradient columns must match input columns");
        }

        Matrix attention_grad = self_attention->backward(attn_dropout_grad, attn_normalized, target_distribution);
        if (attention_grad.cols() != input.cols()) {
            throw std::runtime_error("Attention gradient columns must match input columns");
        }

        // Backward through attention layer norm
        Matrix attention_ln_grad = attention_ln->backward(attention_grad, input);
        if (attention_ln_grad.cols() != input.cols()) {
            throw std::runtime_error("Attention layer norm gradient columns must match input columns");
        }

        // Second residual connection - proper gradient flow
        Matrix final_grad = attention_ln_grad + residual_grad;  // Add both gradient paths
        std::cout << "Final grad dimensions: " << final_grad.rows() << "x"
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
    auto init_weights = [](Matrix& weights, float fan_in, float fan_out) {
        float limit = std::sqrt(6.0f / (fan_in + fan_out));
        limit = std::min(limit, 0.1f); // Cap maximum initialization value
        
        for (size_t i = 0; i < weights.rows(); i++) {
            for (size_t j = 0; j < weights.cols(); j++) {
                weights(i, j) = (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * limit;
            }
        }
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

    // Initialize final layer norm weights (scale and bias)
    if (final_ln) {
        auto& params = final_ln->parameters();
        params.gamma.initialize_constant(1.0f);  // Identity scaling
        params.beta.initialize_constant(0.0f);   // No initial shift
        
        // Cache the initial values for gradient computation
        GradientCheckpoint::cache_activation("final_ln_gamma", params.gamma);
        GradientCheckpoint::cache_activation("final_ln_beta", params.beta);
    }
}

// Add new BatchSequence structure to handle proper sequence boundaries
struct BatchSequence {
    Matrix embeddings;          // [batch_size x seq_len x hidden_size]
    Matrix attention_mask;      // [batch_size x seq_len x seq_len]
    std::vector<size_t> lengths;  // Original sequence lengths
};

Matrix Transformer::forward(const std::vector<int>& input_tokens, const std::string& original_query, const Tokenizer& tokenizer) {
    // Reshape input into proper batch structure
    const size_t batch_size = 1;  // For single sequence input
    const size_t seq_len = input_tokens.size();
    
    // Create proper 3D tensors for batched sequence processing
    BatchSequence batch;
    batch.lengths.push_back(seq_len);
    
    // Get embeddings with proper shape [batch_size x seq_len x hidden_size]
    Matrix token_embeds = token_embedding->forward(input_tokens);
    
    // Create position indices matrix for positional encoding
    Matrix position_ids(seq_len, 1);
    for (size_t i = 0; i < seq_len; i++) {
        position_ids(i, 0) = static_cast<float>(i);
    }
    Matrix pos_embeds = pos_encoding->forward(position_ids);
    
    // Combine embeddings while maintaining batch dimension
    batch.embeddings = Matrix(batch_size * seq_len, config.hidden_size);
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < config.hidden_size; j++) {
            batch.embeddings(i, j) = token_embeds(i, j) + pos_embeds(i, j);
        }
    }
    
    // Create proper causal attention mask [batch_size x seq_len x seq_len]
    batch.attention_mask = Matrix(seq_len, seq_len, 0.0f);
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j <= i; j++) {  // Causal masking
            batch.attention_mask(i, j) = 1.0f;
        }
    }
    
    // Create AttentionMask from Matrix
    AttentionMask attn_mask(batch.attention_mask);
    
    // Apply dropout to embeddings if in training
    if (training && dropout) {
        batch.embeddings = dropout->forward(batch.embeddings, true);
    }
    
    // Process through transformer layers while maintaining sequence structure
    hidden_states = batch.embeddings;
    m_layer_activations.clear();
    m_layer_activations.reserve(layers.size());
    
    // Store sequence boundaries for each layer
    std::vector<std::pair<size_t, size_t>> seq_boundaries;
    seq_boundaries.push_back({0, seq_len});
    
    // Forward through layers with proper sequence handling
    for (size_t i = 0; i < layers.size(); i++) {
        if (!layers[i]) continue;
        
        // Cache input for gradient checkpointing
        std::string layer_key = "layer_" + std::to_string(i) + "_input";
        GradientCheckpoint::cache_activation(layer_key, hidden_states);
        
        // Process through layer with proper attention masking
        hidden_states = layers[i]->forward(hidden_states, attn_mask);
        
        if (training) {
            m_layer_activations.push_back(hidden_states);
        }
    }
    
    // Apply final layer norm
    if (final_ln) {
        hidden_states = final_ln->forward(hidden_states);
    }
    
    // Store sequence info for backward pass
    last_seq_boundaries = seq_boundaries;
    last_input_tokens_ = input_tokens;
    last_input_query_ = original_query;
    
    return hidden_states;
}

void Transformer::clear_kv_cache() {
    for (auto& cache : m_kv_caches) {
        cache.clear();
    }
}

// Original backward method implementation
void Transformer::backward(const Matrix& grad_output, const std::vector<int>& input_tokens, float learning_rate) {
    if (!training) {
        std::cout << "Model is in evaluation mode. Skipping backward pass." << std::endl;
        return;
    }

    try {
        Matrix current_grad = grad_output;
        const float global_max_grad_norm = 5.0f;  // Less aggressive clipping
        static DynamicLossScaler loss_scaler;  // Create static loss scaler
        
        std::cout << "Starting backward pass with " << layers.size() << " layers" << std::endl;
        
        // If using FP16, apply loss scaling
        if (config.use_fp16) {
            float scale = loss_scaler.get_scale();
            current_grad *= scale;
            std::cout << "Applied loss scale: " << scale << std::endl;
        }
        
        // Store layer gradients for global norm computation
        std::vector<Matrix> layer_gradients;
        layer_gradients.reserve(layers.size());
        
        bool has_inf_nan = false;
        
        // First backward pass to compute gradients
        for (int i = layers.size() - 1; i >= 0; --i) {
            std::string ffn_key = "ffn_norm_" + std::to_string(i);
            std::string attn_key = "attn_norm_" + std::to_string(i);
            
            Matrix ffn_input = GradientCheckpoint::get_activation(ffn_key);
            Matrix attn_input = GradientCheckpoint::get_activation(attn_key);
            
            try {
                Matrix layer_grad = layers[i]->backward(current_grad, attn_input, Matrix());
                if (!layer_grad.empty()) {
                    // Check for inf/nan if using FP16
                    if (config.use_fp16 && loss_scaler.has_inf_or_nan(layer_grad)) {
                        has_inf_nan = true;
                        break;
                    }
                    layer_gradients.push_back(layer_grad);
                    current_grad = layer_grad;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error in layer " << i << " backward pass: " << e.what() << std::endl;
                std::cerr << "Attempting to proceed with remaining layers..." << std::endl;
            }
        }
        
        // If using FP16, handle loss scaling
        if (config.use_fp16) {
            bool should_skip = !loss_scaler.update_scale(has_inf_nan);
            if (should_skip) {
                std::cout << "Skipping step due to inf/nan values" << std::endl;
                return;  // Skip this step
            }
            
            // Unscale gradients
            float inv_scale = 1.0f / loss_scaler.get_scale();
            for (auto& grad : layer_gradients) {
                grad *= inv_scale;
            }
        }
        
        // Compute global gradient norm
        float total_grad_norm = 0.0f;
        for (const auto& grad : layer_gradients) {
            #pragma omp parallel for reduction(+:total_grad_norm)
            for (size_t j = 0; j < grad.size(); ++j) {
                total_grad_norm += grad.data()[j] * grad.data()[j];
            }
        }
        total_grad_norm = std::sqrt(total_grad_norm);
        
        std::cout << "Total gradient norm before clipping: " << total_grad_norm << std::endl;
        
        // Compute global scaling factor
        float global_scale = std::min(1.0f, global_max_grad_norm / (total_grad_norm + 1e-6f));
        std::cout << "Global gradient scaling factor: " << global_scale << std::endl;
        
        // Apply scaled gradients and update parameters
        current_grad = grad_output;
        if (config.use_fp16) {
            current_grad *= 1.0f / loss_scaler.get_scale();  // Unscale the gradient
        }
        
        for (int i = layers.size() - 1; i >= 0; --i) {
            std::string ffn_key = "ffn_norm_" + std::to_string(i);
            std::string attn_key = "attn_norm_" + std::to_string(i);
            
            Matrix ffn_input = GradientCheckpoint::get_activation(ffn_key);
            Matrix attn_input = GradientCheckpoint::get_activation(attn_key);
            
            try {
                // Scale the gradient
                current_grad *= global_scale;
                
                Matrix layer_grad = layers[i]->backward(current_grad, attn_input, Matrix());
                if (!layer_grad.empty()) {
                    current_grad = layer_grad;
                    
                    // Update layer parameters with the scaled gradients
                    if (auto* attention = layers[i]->getAttention()) {
                        update_attention_parameters(attention, learning_rate, config);
                    }
                    if (auto* ffn = layers[i]->getFeedForward()) {
                        update_ffn_parameters(ffn, learning_rate, config);
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Error in layer " << i << " parameter update: " << e.what() << std::endl;
            }
        }
        
        std::cout << "Backward pass complete" << std::endl;
        std::cout << "Final gradient norm after scaling: " << (total_grad_norm * global_scale) << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in Transformer backward: " << e.what() << std::endl;
        throw;
    }
}

void Transformer::clear_gradients() {
    // Reset layer gradients
    for (auto& layer : layers) {
        if (auto* attention = layer->getAttention()) {
            attention->param_gradients() = MultiHeadAttention::Gradients();
        }
        if (auto* ffn = layer->getFeedForward()) {
            ffn->param_gradients() = FeedForward::Gradients();
        }
        if (auto* ln = layer->getLayerNorm()) {
            ln->param_gradients() = LayerNorm::Gradients();
        }
    }
    
    // Reset embedding gradients
    if (token_embedding) {
        token_embedding->get_gradient_table().initialize_constant(0.0f);
    }
    
    // Reset parameter gradients
    parameter_grads.reset();
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
    static int current_step = 0;
    const int gradient_accumulation_steps = 4;  // Accumulate over 4 steps
    static LearningRateScheduler lr_scheduler;  // Use our new scheduler
    
    try {
        // Clear accumulated gradients at the start of accumulation
        if (current_step % gradient_accumulation_steps == 0) {
            clear_gradients();
        }
        
        // Compute batch size for proper scaling
        size_t effective_batch_size = input_tokens.size();
        float scale_factor = 1.0f / (gradient_accumulation_steps * effective_batch_size);
        
        // Forward and backward passes with gradient accumulation
        std::vector<Matrix> batch_outputs;
        float total_loss = 0.0f;
        
        for (size_t i = 0; i < input_tokens.size(); ++i) {
            // Forward pass
            Matrix output = forward(input_tokens[i], "", tokenizer);
            batch_outputs.push_back(output);
            
            // Compute loss
            float batch_loss = compute_loss(output, target_distribution);
            total_loss += batch_loss;
            
            // Scale gradients for accumulation
            Matrix scaled_grad = compute_loss_gradient(output, target_distribution);
            scaled_grad *= scale_factor;
            
            // Backward pass with scaled gradients
            backward(scaled_grad, input_tokens[i], 0.0f);  // Don't apply updates yet
        }
        
        // Update parameters only after accumulating gradients
        if ((current_step + 1) % gradient_accumulation_steps == 0) {
            float current_lr = lr_scheduler.get_lr(current_step / gradient_accumulation_steps);
            std::cout << "Step " << current_step / gradient_accumulation_steps 
                      << ", Learning Rate: " << current_lr << std::endl;
            
            // Apply accumulated gradients
            update_parameters(current_lr);
            
            // Log training statistics
            std::cout << "Average Loss: " << (total_loss / effective_batch_size) << std::endl;
        }
        
        current_step++;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in train_step: " << e.what() << std::endl;
        throw;
    }
}

void Transformer::update_parameters(float learning_rate) {
    std::cout << "=== Transformer::update_parameters START ===" << std::endl;

    // Update layer parameters
    for (auto& layer : layers) {
        // Update attention parameters
        if (auto* attention = layer->getAttention()) {
            auto& attn_params = attention->parameters();
            auto& attn_grads = attention->param_gradients();

            // Update weights
            ::update_parameter_with_clip(attn_params.query_weights, attn_grads.query_grad, learning_rate, config);
            ::update_parameter_with_clip(attn_params.key_weights, attn_grads.key_grad, learning_rate, config);
            ::update_parameter_with_clip(attn_params.value_weights, attn_grads.value_grad, learning_rate, config);
            ::update_parameter_with_clip(attn_params.output_weights, attn_grads.output_grad, learning_rate, config);

            // Update biases
            ::update_parameter_with_clip(attn_params.query_bias, attn_grads.query_bias_grad, learning_rate, config);
            ::update_parameter_with_clip(attn_params.key_bias, attn_grads.key_bias_grad, learning_rate, config);
            ::update_parameter_with_clip(attn_params.value_bias, attn_grads.value_bias_grad, learning_rate, config);
            ::update_parameter_with_clip(attn_params.output_bias, attn_grads.output_bias_grad, learning_rate, config);
        }

        // Update feed forward parameters
        if (auto* ffn = layer->getFeedForward()) {
            auto& ffn_params = ffn->parameters();
            auto& ffn_grads = ffn->param_gradients();

            // Update weights
            ::update_parameter_with_clip(ffn_params.ff1_weights, ffn_grads.ff1_grad, learning_rate, config);
            ::update_parameter_with_clip(ffn_params.ff2_weights, ffn_grads.ff2_grad, learning_rate, config);

            // Update biases
            ::update_parameter_with_clip(ffn_params.ff1_bias, ffn_grads.ff1_bias_grad, learning_rate, config);
            ::update_parameter_with_clip(ffn_params.ff2_bias, ffn_grads.ff2_bias_grad, learning_rate, config);
        }

        // Update layer norm parameters
        if (auto* ln = layer->getLayerNorm()) {
            auto& ln_params = ln->parameters();
            auto& ln_grads = ln->param_gradients();
            
            ::update_parameter_with_clip(ln_params.gamma, ln_grads.gamma_grad, learning_rate, config);
            ::update_parameter_with_clip(ln_params.beta, ln_grads.beta_grad, learning_rate, config);
        }
    }

    std::cout << "=== Transformer::update_parameters END ===" << std::endl;
}

std::vector<Matrix>& Transformer::parameters() {
    static std::vector<Matrix> all_params;
    all_params.clear();

    // Token embedding parameters
    if (token_embedding) {
        all_params.push_back(token_embedding->get_weights());
    }

    // Layer parameters
    for (const auto& layer : layers) {
        if (auto* attention = layer->getAttention()) {
            auto& attn_params = attention->parameters();
            all_params.push_back(attn_params.query_weights);
            all_params.push_back(attn_params.key_weights);
            all_params.push_back(attn_params.value_weights);
            all_params.push_back(attn_params.output_weights);
        }

        if (auto* ffn = layer->getFeedForward()) {
            auto& ffn_params = ffn->parameters();
            all_params.push_back(ffn_params.ff1_weights);
            all_params.push_back(ffn_params.ff2_weights);
        }
    }

    return all_params;
}

void Transformer::initialize_weights() {
    // Xavier/Glorot initialization with proper scaling and bounds
    auto init_weights = [](Matrix& weights, size_t fan_in, size_t fan_out, bool is_attention = false) {
        float scale = std::sqrt(2.0f / (fan_in + fan_out));
        if (is_attention) {
            // Attention weights need smaller initialization
            scale *= 0.1f;
        }
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, scale);
        
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < weights.rows(); i++) {
            for (size_t j = 0; j < weights.cols(); j++) {
                float value = dist(gen);
                // Clip extreme values
                value = std::max(-2.0f * scale, std::min(2.0f * scale, value));
                weights(i, j) = value;
            }
        }
    };
    
    // Initialize token embeddings with smaller scale
    if (token_embedding) {
        auto& embedding_weights = token_embedding->get_weights();
        init_weights(embedding_weights, config.vocab_size, config.hidden_size);
        embedding_weights *= 0.1f;  // Scale down embeddings
    }
    
    // Initialize transformer layers
    for (auto& layer : layers) {
        if (layer) {
            // Initialize attention weights
            if (auto* attention = layer->getAttention()) {
                auto& params = attention->parameters();
                
                // Query, Key, Value weights need careful initialization
                init_weights(params.query_weights, config.hidden_size, config.hidden_size, true);
                init_weights(params.key_weights, config.hidden_size, config.hidden_size, true);
                init_weights(params.value_weights, config.hidden_size, config.hidden_size, true);
                init_weights(params.output_weights, config.hidden_size, config.hidden_size, true);
                
                // Initialize attention biases to zero
                params.query_bias.initialize_constant(0.0f);
                params.key_bias.initialize_constant(0.0f);
                params.value_bias.initialize_constant(0.0f);
                params.output_bias.initialize_constant(0.0f);
            }
            
            // Initialize feed forward weights
            if (auto* ff = layer->getFeedForward()) {
                auto& params = ff->parameters();
                
                // FF1 (expansion) and FF2 (projection) weights
                init_weights(params.ff1_weights, config.hidden_size, config.intermediate_size);
                init_weights(params.ff2_weights, config.intermediate_size, config.hidden_size);
                
                // Initialize FF biases to small positive values for ReLU
                params.ff1_bias.initialize_constant(0.01f);
                params.ff2_bias.initialize_constant(0.01f);
            }
            
            // Initialize layer norm parameters
            if (auto* ln = layer->getLayerNorm()) {
                auto& params = ln->parameters();
                params.gamma.initialize_constant(1.0f);  // Identity scaling
                params.beta.initialize_constant(0.0f);   // No initial shift
            }
        }
    }
    
    // Initialize final layer norm
    if (final_ln) {
        auto& params = final_ln->parameters();
        params.gamma.initialize_constant(1.0f);
        params.beta.initialize_constant(0.0f);
    }
    
    std::cout << "Weights initialized with proper scaling" << std::endl;
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

void update_parameter_with_clip(Matrix& param, const Matrix& grad, float learning_rate, const TransformerConfig& config) {
    const float clip_threshold = 1.0f;  // Gradient clipping threshold
    const float weight_decay = config.weight_decay;  // Get weight decay from config
    
    // Calculate gradient norm
    float grad_norm = 0.0f;
    #pragma omp parallel for reduction(+:grad_norm)
    for (size_t i = 0; i < grad.rows(); ++i) {
        for (size_t j = 0; j < grad.cols(); ++j) {
            grad_norm += grad(i, j) * grad(i, j);
        }
    }
    grad_norm = std::sqrt(grad_norm);
    
    // Apply clipping if necessary
    float scaling_factor = grad_norm > clip_threshold ? clip_threshold / grad_norm : 1.0f;
    
    // Update parameters with clipped gradients and weight decay
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < param.rows(); ++i) {
        for (size_t j = 0; j < param.cols(); ++j) {
            float decay = weight_decay * param(i, j);
            param(i, j) -= learning_rate * (grad(i, j) * scaling_factor + decay);
        }
    }
}

void update_parameter_with_clip(Vector& param, const Vector& grad, float learning_rate, const TransformerConfig& config) {
    const float clip_threshold = 1.0f;  // Gradient clipping threshold
    const float weight_decay = config.weight_decay;  // Get weight decay from config
    
    // Calculate gradient norm
    float grad_norm = 0.0f;
    #pragma omp parallel for reduction(+:grad_norm)
    for (size_t i = 0; i < grad.size(); ++i) {
        grad_norm += grad[i] * grad[i];
    }
    grad_norm = std::sqrt(grad_norm);
    
    // Apply clipping if necessary
    float scaling_factor = grad_norm > clip_threshold ? clip_threshold / grad_norm : 1.0f;
    
    // Update parameters with clipped gradients and weight decay
    #pragma omp parallel for
    for (size_t i = 0; i < param.size(); ++i) {
        float decay = weight_decay * param[i];
        param[i] -= learning_rate * (grad[i] * scaling_factor + decay);
    }
}

float compute_loss(const Matrix& output, const Matrix& target_distribution) {
    if (output.size() != target_distribution.size()) {
        throw std::runtime_error("Output and target distribution must have the same size");
    }

    const size_t batch_size = output.rows();
    const size_t vocab_size = output.cols();
    float total_loss = 0.0f;

    // Compute cross-entropy loss for each item in the batch
    #pragma omp parallel for reduction(+:total_loss)
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < vocab_size; ++j) {
            if (target_distribution(i, j) > 0.0f) {  // Only compute loss for actual targets
                // Add small epsilon to prevent log(0)
                const float epsilon = 1e-10f;
                float pred = std::clamp(output(i, j), epsilon, 1.0f - epsilon);
                total_loss -= target_distribution(i, j) * std::log(pred);
            }
        }
    }

    // Average the loss over the batch
    float avg_loss = total_loss / static_cast<float>(batch_size);

    // Check for NaN or Inf
    if (!std::isfinite(avg_loss)) {
        throw std::runtime_error("Loss computation resulted in non-finite value");
    }

    return avg_loss;
}

Matrix compute_loss_gradient(const Matrix& output, const Matrix& target_distribution) {
    if (output.size() != target_distribution.size()) {
        throw std::runtime_error("Output and target distribution must have the same size");
    }

    const size_t batch_size = output.rows();
    const size_t vocab_size = output.cols();
    Matrix gradient(batch_size, vocab_size);

    // Compute gradient of cross-entropy loss with numerical stability
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < vocab_size; ++j) {
            if (target_distribution(i, j) > 0.0f) {
                // Add small epsilon to prevent division by zero
                const float epsilon = 1e-10f;
                float pred = std::clamp(output(i, j), epsilon, 1.0f - epsilon);
                // Gradient for cross-entropy loss: (prediction - target) / (prediction * (1 - prediction))
                gradient(i, j) = (pred - target_distribution(i, j)) / (pred * (1.0f - pred));
            } else {
                gradient(i, j) = 0.0f;  // No gradient for padding/unused tokens
            }
        }
    }

    // Check for NaN or Inf in gradients
    bool has_bad_values = false;
    #pragma omp parallel for collapse(2) reduction(||:has_bad_values)
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < vocab_size; ++j) {
            has_bad_values = has_bad_values || !std::isfinite(gradient(i, j));
        }
    }

    if (has_bad_values) {
        throw std::runtime_error("Gradient computation resulted in non-finite values");
    }

    return gradient;
}

// Add helper function declarations
void update_attention_parameters(MultiHeadAttention* attention, float learning_rate, const TransformerConfig& config) {
    auto& params = attention->parameters();
    auto& grads = attention->param_gradients();
    
    // Update query parameters
    update_parameter_with_clip(params.query_weights, grads.query_grad, learning_rate, config);
    update_parameter_with_clip(params.query_bias, grads.query_bias_grad, learning_rate, config);
    
    // Update key parameters
    update_parameter_with_clip(params.key_weights, grads.key_grad, learning_rate, config);
    update_parameter_with_clip(params.key_bias, grads.key_bias_grad, learning_rate, config);
    
    // Update value parameters
    update_parameter_with_clip(params.value_weights, grads.value_grad, learning_rate, config);
    update_parameter_with_clip(params.value_bias, grads.value_bias_grad, learning_rate, config);
    
    // Update output parameters
    update_parameter_with_clip(params.output_weights, grads.output_grad, learning_rate, config);
    update_parameter_with_clip(params.output_bias, grads.output_bias_grad, learning_rate, config);
}

void update_ffn_parameters(FeedForward* ffn, float learning_rate, const TransformerConfig& config) {
    auto& params = ffn->parameters();
    auto& grads = ffn->param_gradients();
    
    // Update FF1 parameters
    update_parameter_with_clip(params.ff1_weights, grads.ff1_grad, learning_rate, config);
    update_parameter_with_clip(params.ff1_bias, grads.ff1_bias_grad, learning_rate, config);
    
    // Update FF2 parameters
    update_parameter_with_clip(params.ff2_weights, grads.ff2_grad, learning_rate, config);
    update_parameter_with_clip(params.ff2_bias, grads.ff2_bias_grad, learning_rate, config);
}

// Add as a static member of the Transformer class
static DynamicLossScaler loss_scaler;

// Add learning_rate parameter to backward_pass
void Transformer::backward_pass(const Matrix& output, const Matrix& target_distribution, float learning_rate) {
    // Compute loss gradient with proper sequence handling
    Matrix loss_grad = compute_loss_gradient(output, target_distribution);
    
    // Retrieve sequence boundaries
    const auto& seq_boundaries = last_seq_boundaries;
    
    // Apply loss scaling if using FP16
    if (config.use_fp16) {
        float scale = loss_scaler.get_scale();
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < loss_grad.rows(); i++) {
            for (size_t j = 0; j < loss_grad.cols(); j++) {
                loss_grad(i, j) *= scale;
            }
        }
    }
    
    // Initialize gradient accumulation buffers
    std::vector<Matrix> layer_gradients;
    layer_gradients.reserve(layers.size());
    
    bool has_inf_nan = false;
    
    // Backward through layers with proper sequence handling
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        // Get cached layer input
        std::string layer_key = "layer_" + std::to_string(i) + "_input";
        Matrix layer_input = GradientCheckpoint::get_activation(layer_key);
        
        // Process each sequence in the batch separately
        Matrix layer_grad(layer_input.rows(), layer_input.cols(), 0.0f);
        
        for (const auto& [start, end] : seq_boundaries) {
            // Extract sequence portion of gradients and input
            Matrix seq_grad = loss_grad.block(start, 0, end - start, loss_grad.cols());
            Matrix seq_input = layer_input.block(start, 0, end - start, layer_input.cols());
            
            // Backward through layer for this sequence
            Matrix seq_layer_grad = layers[i]->backward(seq_grad, seq_input, target_distribution);
            
            // Accumulate gradients
            for (size_t r = 0; r < seq_layer_grad.rows(); r++) {
                for (size_t c = 0; c < seq_layer_grad.cols(); c++) {
                    layer_grad(start + r, c) += seq_layer_grad(r, c);
                }
            }
        }
        
        // Check for inf/nan if using FP16
        if (config.use_fp16) {
            has_inf_nan |= loss_scaler.has_inf_or_nan(layer_grad);
            
            // Check layer gradients
            auto& layer = layers[i];
            if (auto* attention = layer->getAttention()) {
                auto& grads = attention->param_gradients();
                has_inf_nan |= loss_scaler.has_inf_or_nan(grads.query_grad);
                has_inf_nan |= loss_scaler.has_inf_or_nan(grads.key_grad);
                has_inf_nan |= loss_scaler.has_inf_or_nan(grads.value_grad);
            }
            if (auto* ffn = layer->getFeedForward()) {
                auto& grads = ffn->param_gradients();
                has_inf_nan |= loss_scaler.has_inf_or_nan(grads.ff1_grad);
                has_inf_nan |= loss_scaler.has_inf_or_nan(grads.ff2_grad);
            }
        }
        
        layer_gradients.push_back(layer_grad);
        loss_grad = layer_grad;  // Propagate gradients to next layer
    }
    
    // Handle FP16 loss scaling
    if (config.use_fp16) {
        bool should_skip = !loss_scaler.update_scale(has_inf_nan);
        if (should_skip) {
            std::cout << "Skipping step due to inf/nan in gradients" << std::endl;
            return;
        }
        
        // Unscale gradients
        float inv_scale = 1.0f / loss_scaler.get_scale();
        for (auto& grad : layer_gradients) {
            grad *= inv_scale;
        }
        
        // Unscale parameter gradients
        for (auto& layer : layers) {
            if (auto* attention = layer->getAttention()) {
                auto& grads = attention->param_gradients();
                unscale_gradients(grads, inv_scale);
            }
            if (auto* ffn = layer->getFeedForward()) {
                auto& grads = ffn->param_gradients();
                unscale_gradients(grads, inv_scale);
            }
        }
    }
    
    // Update parameters with proper sequence handling
    for (size_t i = 0; i < layers.size(); i++) {
        auto& layer = layers[i];
        if (auto* attention = layer->getAttention()) {
            update_attention_parameters(attention, learning_rate, config);
        }
        if (auto* ffn = layer->getFeedForward()) {
            update_ffn_parameters(ffn, learning_rate, config);
        }
    }
    
    // Store sequence boundaries for next backward pass
    last_seq_boundaries = seq_boundaries;
}

// Helper function to unscale gradients
void Transformer::unscale_gradients(MultiHeadAttention::Gradients& grads, float scale) {
    // Unscale weight gradients
    grads.query_grad *= scale;
    grads.key_grad *= scale;
    grads.value_grad *= scale;
    grads.output_grad *= scale;

    // Unscale bias gradients
    grads.query_bias_grad *= scale;
    grads.key_bias_grad *= scale;
    grads.value_bias_grad *= scale;
    grads.output_bias_grad *= scale;
}

void Transformer::unscale_gradients(FeedForward::Gradients& grads, float scale) {
    // Unscale weight gradients
    grads.ff1_grad *= scale;
    grads.ff2_grad *= scale;

    // Unscale bias gradients
    grads.ff1_bias_grad *= scale;
    grads.ff2_bias_grad *= scale;
}