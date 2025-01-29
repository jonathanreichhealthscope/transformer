#include "../include/lm_head.hpp"
#include "../include/token_constants.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <random>
#include <deque>
#include <cassert>

// Add minimum active tokens constant
constexpr size_t MIN_ACTIVE_TOKENS = 1000;  // Reasonable default value

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

LanguageModelHead::LanguageModelHead(size_t hidden_size, size_t vocab_size)
    : hidden_size_(hidden_size), vocab_size_(vocab_size), projection(hidden_size, vocab_size),
      bias(vocab_size, 0.0f), token_frequencies(vocab_size, 0.0f), pruning_threshold(1e-6f),
      active_tokens(vocab_size, 1), training_steps(0), is_training_(false),
      // Initialize Adam optimizer state
      m_proj(hidden_size, vocab_size, 0.0f),
      v_proj(hidden_size, vocab_size, 0.0f),
      m_bias(vocab_size, 0.0f),
      v_bias(vocab_size, 0.0f),
      t(0),
      beta1(0.9f),
      beta2(0.999f),
      eps(1e-8f),
      // Initialize learning rate parameters
      current_lr(0.001f),
      min_lr(0.0001f),
      max_lr(0.01f),
      lr_decay(0.99f),
      lr_growth(1.1f),
      // Initialize layer norm
      layer_norm(std::make_unique<LayerNorm>(hidden_size))
{
    // Initialize with larger scale for projection matrix
    float scale = std::sqrt(6.0f / hidden_size);  // Increased initialization scale
    std::cout << "Initializing projection matrix with scale: " << scale << std::endl;
    projection.randomize(-scale, scale);
    
    // Verify initialization
    float min_proj = std::numeric_limits<float>::infinity();
    float max_proj = -std::numeric_limits<float>::infinity();
    float sum_proj = 0.0f;
    size_t nonzero_proj = 0;
    
    #pragma omp parallel for collapse(2) reduction(min:min_proj) reduction(max:max_proj) \
                             reduction(+:sum_proj,nonzero_proj)
    for (size_t i = 0; i < projection.rows(); ++i) {
        for (size_t j = 0; j < projection.cols(); ++j) {
            float val = projection(i, j);
            min_proj = std::min(min_proj, val);
            max_proj = std::max(max_proj, val);
            sum_proj += val;
            if (std::abs(val) > 1e-6) nonzero_proj++;
        }
    }
    
    std::cout << "Initial Projection Matrix Statistics:\n"
              << "Min proj: " << min_proj << "\n"
              << "Max proj: " << max_proj << "\n"
              << "Mean proj: " << sum_proj / (projection.rows() * projection.cols()) << "\n"
              << "Nonzero proj: " << nonzero_proj << "/" 
              << (projection.rows() * projection.cols()) << "\n\n";
    
    // Initialize bias with a slight preference for common tokens
    #pragma omp parallel for
    for (size_t i = 0; i < vocab_size; i++) {
        // Special tokens get neutral bias
        if (i < 5) {
            bias[i] = 0.0f;
        }
        // Common tokens (first ~1000) get slight positive bias
        else if (i < 1000) {
            bias[i] = 0.1f;
        }
        // Rest get slight negative bias
        else {
            bias[i] = -0.1f;
        }
    }
    
    // Initialize token frequencies with meaningful values
    #pragma omp parallel for
    for (size_t i = 0; i < vocab_size; i++) {
        if (i < 5) {  // Special tokens
            token_frequencies[i] = 0.5f;  // Moderate frequency
        }
        else if (i < 1000) {  // Common tokens
            token_frequencies[i] = 1.0f - (i / 1000.0f);  // Gradually decreasing frequency
        }
        else {  // Less common tokens
            token_frequencies[i] = 0.1f;  // Small but non-zero frequency
        }
    }
    
    active_token_indices.reserve(vocab_size);
    #pragma omp parallel for
    for (size_t i = 0; i < vocab_size; i++) {
        active_token_indices.push_back(i);
    }
    
    std::cout << "Initializing LM Head with vocab size " << vocab_size 
              << " and hidden size " << hidden_size << std::endl;
}

Matrix LanguageModelHead::forward(const Matrix& hidden_states, bool training) {
    // Apply layer normalization first
    Matrix normalized = layer_norm->forward(hidden_states);
    
    // Project to vocabulary size using projection matrix directly
    Matrix logits = matmul(normalized, projection);
    
    // Add bias terms
    for (size_t i = 0; i < logits.rows(); ++i) {
        for (size_t j = 0; j < logits.cols(); ++j) {
            logits(i, j) += bias[j];
        }
    }
    
    // Use consistent temperature scaling
    const float temperature = 0.8f;  // Single temperature value for both training and inference
    logits = logits * (1.0f / temperature);
    
    // Apply format-specific biasing
    bias_completion_format(logits);
    
    // Clip logits to prevent extreme values
    const float clip_val = 5.0f;
    for (size_t i = 0; i < logits.rows(); ++i) {
        for (size_t j = 0; j < logits.cols(); ++j) {
            logits(i, j) = std::max(-clip_val, std::min(clip_val, logits(i, j)));
        }
    }
    
    return logits;
}

Matrix LanguageModelHead::forward_impl(const Matrix& hidden_states) {
    try {
        // Project hidden states to vocabulary space
        Matrix logits = matmul(hidden_states, projection);
        
        // Add bias with proper token activation
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < logits.rows(); ++i) {
            for (size_t j = 0; j < logits.cols(); ++j) {
                logits(i, j) += bias[j];
            }
        }
        
        // Apply dynamic temperature scaling based on logit distribution
        float max_logit = -std::numeric_limits<float>::infinity();
        float min_logit = std::numeric_limits<float>::infinity();
        
        #pragma omp parallel for collapse(2) reduction(max:max_logit) reduction(min:min_logit)
        for (size_t i = 0; i < logits.rows(); ++i) {
            for (size_t j = 0; j < logits.cols(); ++j) {
                max_logit = std::max(max_logit, logits(i, j));
                min_logit = std::min(min_logit, logits(i, j));
            }
        }
        
        // Dynamic temperature based on logit range
        float logit_range = max_logit - min_logit;
        float dynamic_temp = std::max(0.1f, std::min(2.0f, logit_range / 10.0f));
        
        // Apply temperature scaling
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < logits.rows(); ++i) {
            for (size_t j = 0; j < logits.cols(); ++j) {
                logits(i, j) /= dynamic_temp;
            }
        }
        
        return logits;
    } catch (const std::exception& e) {
        throw std::runtime_error("LMHead forward failed: " + std::string(e.what()));
    }
}

Matrix LanguageModelHead::project_to_vocab(const Matrix& hidden_states) {
    this->hidden_states = hidden_states;
    size_t total_size = hidden_states.rows();
    size_t hidden_dim = hidden_states.cols();
    
    if (hidden_dim != hidden_size_) {
        throw std::runtime_error("Hidden dimension mismatch: " + std::to_string(hidden_dim) +
                               " != " + std::to_string(hidden_size_));
    }
    
    return forward_impl(hidden_states);
}

Matrix LanguageModelHead::backward(const Matrix& grad_output, const Matrix& target_distribution) {
    return backward_pass(grad_output, hidden_states);  // Use the existing backward_pass implementation
}

void LanguageModelHead::backward_linear(const Matrix& grad_output) {
    // Use backward_pass which already has Adam optimization
    backward_pass(grad_output, hidden_states);
}

void LanguageModelHead::update_learning_rate(float current_loss) {
    // Add loss to history
    loss_history.push_back(current_loss);
    if (loss_history.size() > LOSS_HISTORY_SIZE) {
        loss_history.pop_front();
    }
    
    // Only adjust learning rate if we have enough history
    if (loss_history.size() >= 2) {
        float avg_recent_loss = 0.0f;
        float avg_old_loss = 0.0f;
        size_t recent_count = loss_history.size() / 2;
        
        // Calculate average of recent and older losses
        for (size_t i = 0; i < loss_history.size(); i++) {
            if (i >= loss_history.size() - recent_count) {
                avg_recent_loss += loss_history[i];
            } else {
                avg_old_loss += loss_history[i];
            }
        }
        avg_recent_loss /= recent_count;
        avg_old_loss /= (loss_history.size() - recent_count);
        
        // Adjust learning rate based on loss trend
        if (avg_recent_loss < avg_old_loss) {
            // Loss is decreasing, increase learning rate slightly
            current_lr = std::min(max_lr, current_lr * lr_growth);
        } else {
            // Loss is increasing or stagnant, decrease learning rate
            current_lr = std::max(min_lr, current_lr * lr_decay);
        }
    }
    
    prev_loss = current_loss;
}

void LanguageModelHead::update_token_frequencies(const std::vector<int>& tokens) {
    // Reset frequencies periodically to prevent over-accumulation
    if (training_steps % 1000 == 0) {  // Reset every 1000 steps
        #pragma omp parallel for
        for (size_t i = 0; i < token_frequencies.size(); i++) {
            token_frequencies[i] = 0.0f;
        }
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < tokens.size(); i++) {
        int token = tokens[i];
        if (token >= 0 && static_cast<size_t>(token) < vocab_size_) {
            #pragma omp atomic
            token_frequencies[token] += 1.0f;
        }
    }
    training_steps++;
    
    // Normalize frequencies to prevent extreme values
    if (!token_frequencies.empty()) {
        float max_freq = *std::max_element(token_frequencies.begin(), token_frequencies.end());
        if (max_freq > 0) {
            #pragma omp parallel for
            for (size_t i = 0; i < token_frequencies.size(); i++) {
                token_frequencies[i] /= max_freq;  // Normalize to [0,1] range
            }
        }
    }
}

void LanguageModelHead::update_active_tokens() {
    const float decay = 0.99f;
    
    // Parallelize frequency decay
    #pragma omp parallel for
    for (size_t i = 0; i < vocab_size_; i++) {
        token_frequencies[i] *= decay;
    }
    
    size_t active_count = 0;
    active_token_indices.clear();
    
    // Use vector of pairs to avoid multiple passes
    std::vector<std::pair<float, size_t>> freq_pairs(vocab_size_);
    
    #pragma omp parallel for
    for (size_t i = 0; i < vocab_size_; i++) {
        freq_pairs[i] = {token_frequencies[i], i};
    }
    
    // Partial sort only what we need
    std::partial_sort(freq_pairs.begin(), 
                     freq_pairs.begin() + std::min(MIN_ACTIVE_TOKENS, vocab_size_),
                     freq_pairs.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Reset active tokens
    std::fill(active_tokens.begin(), active_tokens.end(), 0);
    active_token_indices.clear();
    active_token_indices.reserve(std::min(MIN_ACTIVE_TOKENS, vocab_size_));
    
    // Set active tokens based on sorted frequencies
    for (size_t i = 0; i < std::min(MIN_ACTIVE_TOKENS, vocab_size_); i++) {
        size_t idx = freq_pairs[i].second;
        active_tokens[idx] = 1;
        active_token_indices.push_back(idx);
    }
}

#ifdef USE_CUDA
// Add the new GPU kernel for FP16 conversion
__global__ void convert_projection_to_fp16_kernel(
    half* output, const float* input, const unsigned char* active_tokens,
    size_t hidden_size, size_t vocab_size) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size * vocab_size && active_tokens[idx / hidden_size]) {
        output[idx] = __float2half(input[idx]);
    }
}
#endif

LanguageModelHead::~LanguageModelHead() {
#ifdef USE_CUDA
    if (cublas_handle) {
        cublasDestroy(cublas_handle);
    }
    if (d_projection) cudaFree(d_projection);
    if (d_bias) cudaFree(d_bias);
    if (d_projection_fp16) cudaFree(d_projection_fp16);
    if (d_hidden_states_fp16) cudaFree(d_hidden_states_fp16);
    if (d_output_fp16) cudaFree(d_output_fp16);
    if (d_output) cudaFree(d_output);
    if (d_active_tokens) cudaFree(d_active_tokens);
    if (d_active_token_indices) cudaFree(d_active_token_indices);
    if (compute_stream) cudaStreamDestroy(compute_stream);
#endif
}

void LanguageModelHead::set_training(bool training_mode) {
    is_training_ = training_mode;
}

Matrix LanguageModelHead::backward_pass(const Matrix& grad_output, const Matrix& hidden_states) {
    // Compute gradients for projection and bias
    std::cout << "Computing gradients for projection and bias" << std::endl;
    Matrix grad_proj = matmul(hidden_states.transpose(), grad_output);
    
    // Gradient clipping for stability
    const float grad_clip = 1.0f;
    for (size_t i = 0; i < grad_proj.rows(); ++i) {
        for (size_t j = 0; j < grad_proj.cols(); ++j) {
            grad_proj(i, j) = std::max(-grad_clip, std::min(grad_clip, grad_proj(i, j)));
        }
    }
    
    Vector grad_bias = grad_output.row_sum();
    for (size_t i = 0; i < grad_bias.size(); ++i) {
        grad_bias[i] = std::max(-grad_clip, std::min(grad_clip, grad_bias[i]));
    }

    t++;  // Increment time step

    // Update projection matrix using Adam optimizer with improved stability
    const float scale_factor = std::sqrt(1.0f / hidden_size_);  // Reduced scale factor
    const float max_update = 0.05f * scale_factor;  // More conservative max update
    
    // Constants for gradient clipping and stability
    const float clip_threshold = 5.0f;  // Reduced from 10.0f
    const float max_allowed_value = 100.0f;
    bool has_unstable_update = false;
    
    // Validate m_proj and v_proj dimensions match projection matrix
    if (m_proj.rows() != projection.rows() || m_proj.cols() != projection.cols() ||
        v_proj.rows() != projection.rows() || v_proj.cols() != projection.cols()) {
        throw std::runtime_error("Momentum matrices dimension mismatch");
    }
    
    // Print dimensions for debugging
    std::cout << "Matrix dimensions in backward pass:\n"
              << "grad_proj: " << grad_proj.rows() << "x" << grad_proj.cols() << "\n"
              << "projection: " << projection.rows() << "x" << projection.cols() << "\n"
              << "m_proj: " << m_proj.rows() << "x" << m_proj.cols() << "\n"
              << "v_proj: " << v_proj.rows() << "x" << v_proj.cols() << "\n";
    
    // Remove debug prints inside OpenMP region to prevent garbled output
    #pragma omp parallel for collapse(2) reduction(|:has_unstable_update)
    for (size_t i = 0; i < grad_proj.rows(); ++i) {
        for (size_t j = 0; j < grad_proj.cols(); ++j) {
            // Bounds check before any access
            if (i >= grad_proj.rows() || j >= grad_proj.cols() ||
                i >= projection.rows() || j >= projection.cols() ||
                i >= m_proj.rows() || j >= m_proj.cols() ||
                i >= v_proj.rows() || j >= v_proj.cols()) {
                #pragma omp critical
                {
                    std::cout << "ERROR: Index out of bounds at (" << i << "," << j << ")\n";
                    std::cout << "Matrix dimensions:\n"
                             << "grad_proj: " << grad_proj.rows() << "x" << grad_proj.cols() << "\n"
                             << "projection: " << projection.rows() << "x" << projection.cols() << "\n";
                }
                has_unstable_update = true;
                continue;
            }

            if (!std::isfinite(grad_proj(i, j))) {
                continue;
            }
            
            // Clip gradient before momentum update
            float clipped_grad = grad_proj(i, j);
            if (std::abs(clipped_grad) > clip_threshold) {
                clipped_grad *= clip_threshold / std::abs(clipped_grad);
            }
            
            // Update momentum
            float new_m = beta1 * m_proj(i, j) + (1 - beta1) * clipped_grad;
            if (!std::isfinite(new_m)) {
                has_unstable_update = true;
                continue;
            }
            m_proj(i, j) = new_m;
            
            // Update RMSprop
            float grad_squared = clipped_grad * clipped_grad;
            float new_v = beta2 * v_proj(i, j) + (1 - beta2) * grad_squared;
            if (!std::isfinite(new_v)) {
                has_unstable_update = true;
                continue;
            }
            v_proj(i, j) = new_v;
            
            // Bias correction
            float m_hat = m_proj(i, j) / (1 - std::pow(beta1, t));
            float v_hat = v_proj(i, j) / (1 - std::pow(beta2, t));
            
            if (!std::isfinite(m_hat) || !std::isfinite(v_hat)) {
                has_unstable_update = true;
                continue;
            }
            
            // Compute update
            float denom = std::sqrt(v_hat) + eps;
            if (denom < eps) denom = eps;
            
            float update = current_lr * m_hat / denom;
            update *= scale_factor;
            
            if (!std::isfinite(update)) {
                has_unstable_update = true;
                continue;
            }
            
            // Hard clip update
            update = std::max(-max_update, std::min(max_update, update));
            
            // Compute proposed new value
            float new_value = projection(i, j) - update;
            
            if (std::abs(new_value) > max_allowed_value) {
                has_unstable_update = true;
                continue;
            }
            
            if (std::isfinite(new_value)) {
                projection(i, j) = new_value;
            }
        }
    }
    
    std::cout << "[DEBUG] Parameter updates completed" << std::endl;
    std::cout << "[DEBUG] Checking for unstable updates" << std::endl;
    
    if (has_unstable_update) {
        std::cout << "[DEBUG] Unstable updates detected, reducing learning rate" << std::endl;
        current_lr *= 0.5f;
        
        std::cout << "[DEBUG] Resetting momentum and RMSprop states" << std::endl;
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < projection.rows(); ++i) {
            for (size_t j = 0; j < projection.cols(); ++j) {
                m_proj(i, j) = 0.0f;
                v_proj(i, j) = 0.0f;
            }
        }
    }
    
    std::cout << "[DEBUG] Update process completed" << std::endl;

    // Debug output for projection matrix after update
    float min_proj_after = std::numeric_limits<float>::infinity();
    float max_proj_after = -std::numeric_limits<float>::infinity();
    float sum_proj_after = 0.0f;
    size_t nonzero_proj_after = 0;

    #pragma omp parallel for collapse(2) reduction(min:min_proj_after) reduction(max:max_proj_after) \
                             reduction(+:sum_proj_after,nonzero_proj_after)
    for (size_t i = 0; i < projection.rows(); ++i) {
        for (size_t j = 0; j < projection.cols(); ++j) {
            float val = projection(i, j);
            min_proj_after = std::min(min_proj_after, val);
            max_proj_after = std::max(max_proj_after, val);
            sum_proj_after += val;
            if (std::abs(val) > 1e-6) nonzero_proj_after++;
        }
    }

    std::cout << "Projection Matrix Statistics After Update:\n"
              << "Min proj: " << min_proj_after << "\n"
              << "Max proj: " << max_proj_after << "\n"
              << "Mean proj: " << sum_proj_after / (projection.rows() * projection.cols()) << "\n"
              << "Nonzero proj: " << nonzero_proj_after << "/" 
              << (projection.rows() * projection.cols()) << "\n\n";

    // If projection matrix has degenerated, reinitialize it
    if (nonzero_proj_after < projection.rows() * projection.cols() / 2) {
        std::cout << "WARNING: Projection matrix has too many zeros. Reinitializing...\n";
        float scale = std::sqrt(6.0f / (hidden_size_ + vocab_size_));
        projection.randomize(-scale, scale);
    }
    std::cout << "Projection matrix reinitialized" << std::endl;
    // Update bias vector using Adam optimizer with similar changes
    #pragma omp parallel for
    for (size_t i = 0; i < bias.size(); ++i) {
        // Update momentum
        m_bias[i] = beta1 * m_bias[i] + (1 - beta1) * grad_bias[i];
        // Update RMSprop with stability
        v_bias[i] = beta2 * v_bias[i] + (1 - beta2) * grad_bias[i] * grad_bias[i];
        
        // Bias correction
        float m_hat = m_bias[i] / (1 - std::pow(beta1, t));
        float v_hat = v_bias[i] / (1 - std::pow(beta2, t));
        
        // Compute update without restrictive bounds
        float update = current_lr * m_hat / (std::sqrt(v_hat) + eps);
        
        // Apply update directly
        bias[i] -= update;
    }
    std::cout << "Bias vector updated" << std::endl;
    // Compute gradient with respect to input
    Matrix grad_input = matmul(grad_output, projection.transpose());
    if (grad_input.cols() != hidden_states.cols()) {
        throw std::runtime_error("Language model head gradient output dimension (" +
                                 std::to_string(grad_input.cols()) +
                                 ") must match hidden size (" +
                                 std::to_string(hidden_states.cols()) + ")");
    }
    return grad_input;
}

void LanguageModelHead::bias_completion_format(Matrix& logits) {
    if (!tokenizer) {
        return;  // Skip biasing if tokenizer is not set
    }

    // Get special token IDs from tokenizer
    const int sep_token_id = tokenizer->get_sep_token_id();
    
    // Get the last predicted token
    int last_token = -1;  // You'll need to track this
    
    // After separator token, boost probability of tokens that commonly start completions
    if (last_token == sep_token_id) {
        // Boost tokens that typically start completions (e.g., space token)
        // This helps enforce the format where completions start with a space
        const float boost_factor = 2.0f;
        for (size_t i = 0; i < logits.rows(); i++) {
            std::string token = tokenizer->decode({static_cast<int>(i)});
            if (!token.empty() && token[0] == ' ') {
                logits.data()[i] *= boost_factor;
            }
        }
    }
} 
