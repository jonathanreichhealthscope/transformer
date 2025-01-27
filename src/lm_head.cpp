#include "../include/lm_head.hpp"
#include "../include/token_constants.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <random>

// Add minimum active tokens constant
constexpr size_t MIN_ACTIVE_TOKENS = 1000;  // Reasonable default value

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

LanguageModelHead::LanguageModelHead(size_t hidden_size, size_t vocab_size)
    : hidden_size_(hidden_size), vocab_size_(vocab_size), projection(hidden_size, vocab_size),
      bias(vocab_size, 0.0f), token_frequencies(vocab_size, 0.0f), pruning_threshold(1e-6f),
      active_tokens(vocab_size, 1), training_steps(0), is_training_(false)
{
    // Initialize with Xavier/Glorot initialization
    float scale = std::sqrt(6.0f / (hidden_size + vocab_size));
    projection.randomize(-scale, scale);
    
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

Matrix LanguageModelHead::forward_impl(const Matrix& hidden_states) {
    size_t total_size = hidden_states.rows();
    size_t hidden_dim = hidden_states.cols();
    
    if (hidden_dim != hidden_size_) {
        throw std::runtime_error("Hidden dimension mismatch: " + std::to_string(hidden_dim) +
                               " != " + std::to_string(hidden_size_));
    }
    
    // Debug hidden states
    float min_hidden = std::numeric_limits<float>::infinity();
    float max_hidden = -std::numeric_limits<float>::infinity();
    float sum_hidden = 0.0f;
    size_t nonzero_hidden = 0;
    
    #pragma omp parallel for collapse(2) reduction(min:min_hidden) reduction(max:max_hidden) \
                             reduction(+:sum_hidden,nonzero_hidden)
    for (size_t i = 0; i < hidden_states.rows(); i++) {
        for (size_t j = 0; j < hidden_states.cols(); j++) {
            float val = hidden_states(i, j);
            min_hidden = std::min(min_hidden, val);
            max_hidden = std::max(max_hidden, val);
            sum_hidden += val;
            if (std::abs(val) > 1e-6) nonzero_hidden++;
        }
    }
    
    std::cout << "\nHidden States Statistics in forward_impl:\n"
              << "Min hidden: " << min_hidden << "\n"
              << "Max hidden: " << max_hidden << "\n"
              << "Mean hidden: " << sum_hidden / (hidden_states.rows() * hidden_states.cols()) << "\n"
              << "Nonzero hidden: " << nonzero_hidden << "/" 
              << (hidden_states.rows() * hidden_states.cols()) << "\n\n";
    
    // Scale hidden states for better gradient flow
    float scale_factor = std::sqrt(1.0f / hidden_size_);  // Changed scaling factor
    Matrix scaled_hidden = hidden_states;
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < scaled_hidden.rows(); i++) {
        for (size_t j = 0; j < scaled_hidden.cols(); j++) {
            scaled_hidden(i, j) *= scale_factor;
        }
    }
    
    // Debug hidden states after scaling
    float min_scaled = std::numeric_limits<float>::infinity();
    float max_scaled = -std::numeric_limits<float>::infinity();
    float sum_scaled = 0.0f;
    
    #pragma omp parallel for collapse(2) reduction(min:min_scaled) reduction(max:max_scaled) \
                             reduction(+:sum_scaled)
    for (size_t i = 0; i < scaled_hidden.rows(); i++) {
        for (size_t j = 0; j < scaled_hidden.cols(); j++) {
            float val = scaled_hidden(i, j);
            min_scaled = std::min(min_scaled, val);
            max_scaled = std::max(max_scaled, val);
            sum_scaled += val;
        }
    }
    
    std::cout << "Scaled Hidden States Statistics:\n"
              << "Min scaled: " << min_scaled << "\n"
              << "Max scaled: " << max_scaled << "\n"
              << "Mean scaled: " << sum_scaled / (scaled_hidden.rows() * scaled_hidden.cols()) << "\n\n";
              
    // Verify projection matrix values
    float min_proj = std::numeric_limits<float>::infinity();
    float max_proj = -std::numeric_limits<float>::infinity();
    float sum_proj = 0.0f;
    
    #pragma omp parallel for collapse(2) reduction(min:min_proj) reduction(max:max_proj) \
                             reduction(+:sum_proj)
    for (size_t i = 0; i < projection.rows(); i++) {
        for (size_t j = 0; j < projection.cols(); j++) {
            float val = projection(i, j);
            min_proj = std::min(min_proj, val);
            max_proj = std::max(max_proj, val);
            sum_proj += val;
        }
    }
    
    std::cout << "Projection Matrix Statistics:\n"
              << "Min proj: " << min_proj << "\n"
              << "Max proj: " << max_proj << "\n"
              << "Mean proj: " << sum_proj / (projection.rows() * projection.cols()) << "\n\n";
    
    // Compute logits with verification
    Matrix logits = matmul(scaled_hidden, projection);
    
    // Verify logits immediately after matmul
    float min_raw_logit = std::numeric_limits<float>::infinity();
    float max_raw_logit = -std::numeric_limits<float>::infinity();
    float sum_raw_logit = 0.0f;
    
    #pragma omp parallel for collapse(2) reduction(min:min_raw_logit) reduction(max:max_raw_logit) \
                             reduction(+:sum_raw_logit)
    for (size_t i = 0; i < logits.rows(); i++) {
        for (size_t j = 0; j < logits.cols(); j++) {
            float val = logits(i, j);
            min_raw_logit = std::min(min_raw_logit, val);
            max_raw_logit = std::max(max_raw_logit, val);
            sum_raw_logit += val;
        }
    }
    
    std::cout << "Raw Logits Statistics (before bias):\n"
              << "Min raw logit: " << min_raw_logit << "\n"
              << "Max raw logit: " << max_raw_logit << "\n"
              << "Mean raw logit: " << sum_raw_logit / (logits.rows() * logits.cols()) << "\n\n";
              
    // If logits are all zero, reinitialize projection matrix with larger scale
    if (std::abs(max_raw_logit) < 1e-6 && std::abs(min_raw_logit) < 1e-6) {
        std::cout << "WARNING: Logits are all zero. Reinitializing projection matrix...\n";
        float scale = std::sqrt(6.0f / hidden_size_);  // Larger initialization scale
        projection.randomize(-scale, scale);
        logits = matmul(scaled_hidden, projection);  // Recompute logits
    }
    
    // Add bias and apply token-specific adjustments
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < logits.rows(); ++i) {
        for (size_t j = 0; j < logits.cols(); ++j) {
            // Add bias
            logits(i, j) += bias[j];
            
            // Token-specific adjustments
            if (j == static_cast<size_t>(tokens::UNK_ID)) {
                logits(i, j) -= 15.0f;  // Stronger penalty for UNK
            }
            else if (j < 5) {  // Other special tokens
                logits(i, j) -= 5.0f;  // Moderate penalty
            }
            else {
                // Scale logits based on token frequency
                float freq_bonus = token_frequencies[j];
                logits(i, j) += freq_bonus;
            }
        }
    }
    
    // Apply softmax with lower temperature for sharper predictions
    const float temperature = 0.7f;  // Reduced from 0.8f
    #pragma omp parallel for
    for (size_t i = 0; i < logits.rows(); i++) {
        // Find max for numerical stability
        float max_val = logits(i, 0);
        for (size_t j = 1; j < logits.cols(); j++) {
            max_val = std::max(max_val, logits(i, j));
        }
        
        // Apply temperature and compute sum
        float sum = 0.0f;
        for (size_t j = 0; j < logits.cols(); j++) {
            logits(i, j) = std::exp((logits(i, j) - max_val) / temperature);
            sum += logits(i, j);
        }
        
        // Normalize
        if (sum > 1e-6f) {
            for (size_t j = 0; j < logits.cols(); j++) {
                logits(i, j) /= sum;
            }
        }
    }
    
    // Debug output for logit statistics
    float min_logit = std::numeric_limits<float>::infinity();
    float max_logit = -std::numeric_limits<float>::infinity();
    float sum_logits = 0.0f;
    size_t active_logits = 0;
    
    const int bar_width = 50;
    const size_t total_elements = logits.rows() * logits.cols();
    size_t processed_elements = 0;
    
    std::cout << "\nCounting active logits:\n" << std::flush;
    
    for (size_t i = 0; i < logits.rows(); i++) {
        for (size_t j = 0; j < logits.cols(); j++) {
            float val = logits(i, j);
            min_logit = std::min(min_logit, val);
            max_logit = std::max(max_logit, val);
            sum_logits += val;
            if (val > 1e-6) active_logits++;
            
            processed_elements++;
            
            // Update progress bar every 1000 elements for smoother display
            if (processed_elements % 1000 == 0 || processed_elements == total_elements) {
                float progress = float(processed_elements) / total_elements;
                int pos = bar_width * progress;
                
                std::cout << "\r[";
                for (int k = 0; k < bar_width; ++k) {
                    if (k < pos) std::cout << "=";
                    else if (k == pos) std::cout << ">";
                    else std::cout << " ";
                }
                std::cout << "] " << std::fixed << std::setprecision(1) 
                         << (progress * 100.0) << "% "
                         << "(" << processed_elements << "/" << total_elements << ")" 
                         << std::flush;
            }
        }
    }
    std::cout << std::endl << std::endl;  // Add extra newline for spacing
    
    float mean_logit = sum_logits / total_elements;
    float range = max_logit - min_logit;
    
    std::cout << "Logit Statistics in forward_impl:\n"
              << "Min logit: " << std::fixed << std::setprecision(1) << min_logit << "\n"
              << "Max logit: " << max_logit << "\n"
              << "Mean logit: " << mean_logit << "\n"
              << "Range: " << range << "\n"
              << "Active logits: " << active_logits << "/" << total_elements << "\n\n";
    
    return logits;
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
    size_t total_size = hidden_states.rows();
    size_t batch_size = total_size / 2;
    size_t seq_len = 2;
    
    // Scale gradients to match forward pass scaling
    float scale_factor = std::sqrt(2.0f / hidden_size_);  // Match forward pass scaling
    
    Matrix loss_grad(grad_output.rows(), grad_output.cols());
    
    // First compute max logits for numerical stability
    std::vector<float> max_logits(grad_output.rows(), -std::numeric_limits<float>::infinity());
    for (size_t i = 0; i < grad_output.rows(); i++) {
        for (size_t j = 0; j < grad_output.cols(); j++) {
            max_logits[i] = std::max(max_logits[i], grad_output(i, j));
        }
    }
    
    // Compute sum of exponentials for each row
    std::vector<float> sum_exp(grad_output.rows(), 0.0f);
    for (size_t i = 0; i < grad_output.rows(); i++) {
        for (size_t j = 0; j < grad_output.cols(); j++) {
            sum_exp[i] += std::exp(grad_output(i, j) - max_logits[i]);
        }
    }
    
    // Print softmax statistics for debugging
    std::cout << "\nSoftmax Statistics in backward:\n"
              << "Max logit values: " << max_logits[0] << "\n"
              << "Sum of exponentials: " << sum_exp[0] << "\n";
    
    // Compute gradients with proper softmax normalization
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < grad_output.rows(); i++) {
        for (size_t j = 0; j < grad_output.cols(); j++) {
            if (!target_distribution.empty() && target_distribution(i, j) > 0.0f) {
                // Cross entropy gradient with numerical stability
                float softmax = std::exp(grad_output(i, j) - max_logits[i]) / sum_exp[i];
                loss_grad(i, j) = (softmax - target_distribution(i, j)) * scale_factor;
            } else {
                loss_grad(i, j) = grad_output(i, j) * scale_factor;
            }
        }
    }

    // Print gradient statistics
    float min_grad = std::numeric_limits<float>::infinity();
    float max_grad = -std::numeric_limits<float>::infinity();
    float sum_grad = 0.0f;
    size_t nonzero_count = 0;
    for (size_t i = 0; i < loss_grad.rows(); i++) {
        for (size_t j = 0; j < loss_grad.cols(); j++) {
            float val = std::abs(loss_grad(i, j));
            if (val > 1e-10) {
                nonzero_count++;
            }
            min_grad = std::min(min_grad, val);
            max_grad = std::max(max_grad, val);
            sum_grad += val;
        }
    }
    std::cout << "\nGradient Statistics in LM Head backward:\n"
              << "Min gradient: " << min_grad << "\n"
              << "Max gradient: " << max_grad << "\n"
              << "Mean gradient: " << sum_grad / (loss_grad.rows() * loss_grad.cols()) << "\n"
              << "Nonzero gradients: " << nonzero_count << "/" 
              << (loss_grad.rows() * loss_grad.cols()) << "\n";

    Matrix expanded_grad(total_size, vocab_size_);
    expanded_grad.fill(0.0f);
    
    #pragma omp parallel for
    for (size_t b = 0; b < batch_size; b++) {
        size_t src_idx = b;
        size_t dst_idx = (b * seq_len + (seq_len - 1));
        for (size_t v = 0; v < vocab_size_; v++) {
            expanded_grad(dst_idx, v) = loss_grad(src_idx, v);
        }
    }

    backward_linear(expanded_grad);
    
    // Scale gradients for hidden states
    Matrix hidden_grad = matmul(expanded_grad, projection.transpose());
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < hidden_grad.rows(); i++) {
        for (size_t j = 0; j < hidden_grad.cols(); j++) {
            hidden_grad(i, j) *= scale_factor;
        }
    }
    
    return hidden_grad;
}

void LanguageModelHead::backward_linear(const Matrix& grad_output) {
    if (grad_output.rows() != hidden_states.rows()) {
        throw std::runtime_error("Invalid matrix dimensions for gradient computation");
    }

    // Scale hidden states to match forward pass
    Matrix scaled_hidden = hidden_states;
    float scale_factor = std::sqrt(2.0f / hidden_size_);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < scaled_hidden.rows(); i++) {
        for (size_t j = 0; j < scaled_hidden.cols(); j++) {
            scaled_hidden(i, j) *= scale_factor;
        }
    }

    Matrix grad_proj = matmul(scaled_hidden.transpose(), grad_output);
    Vector grad_bias(bias.size(), 0.0f);

    // Compute bias gradients
    #pragma omp parallel
    {
        Vector local_grad_bias(bias.size(), 0.0f);
        #pragma omp for collapse(2)
        for (size_t i = 0; i < grad_output.rows(); i++) {
            for (size_t j = 0; j < grad_output.cols(); j++) {
                local_grad_bias[j] += grad_output(i, j);
            }
        }
        #pragma omp critical
        {
            for (size_t j = 0; j < bias.size(); j++) {
                grad_bias[j] += local_grad_bias[j];
            }
        }
    }

    // Compute gradient statistics for adaptive clipping
    float max_grad = 0.0f;
    float grad_norm = 0.0f;
    
    #pragma omp parallel for reduction(max:max_grad) reduction(+:grad_norm)
    for (size_t i = 0; i < grad_proj.rows(); i++) {
        for (size_t j = 0; j < grad_proj.cols(); j++) {
            float abs_grad = std::abs(grad_proj(i, j));
            max_grad = std::max(max_grad, abs_grad);
            grad_norm += grad_proj(i, j) * grad_proj(i, j);
        }
    }
    grad_norm = std::sqrt(grad_norm);
    
    // Clip gradients using adaptive threshold
    const float clip_threshold = 1.0f;
    float clip_scale = 1.0f;
    if (grad_norm > clip_threshold) {
        clip_scale = clip_threshold / grad_norm;
    }
    
    // Use much smaller base learning rate and decay
    const float base_lr = 0.001f;  // Reduced from 0.01f
    const float min_lr = 0.0001f;  // Minimum learning rate
    float effective_lr = std::max(min_lr, 
                                base_lr / (1.0f + std::sqrt(static_cast<float>(training_steps + 1))));  // Cast to float
    
    // Update projection matrix with safeguards
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < projection.rows(); i++) {
        for (size_t j = 0; j < projection.cols(); j++) {
            float update = effective_lr * clip_scale * grad_proj(i, j);
            
            // Limit maximum update magnitude
            const float max_update = 0.1f;
            if (std::abs(update) > max_update) {
                update = (update > 0 ? max_update : -max_update);
            }
            
            // Update with safeguard against zero values
            float new_value = projection(i, j) - update;
            
            // Prevent values from getting too close to zero
            const float min_abs_value = 1e-4f;
            if (std::abs(new_value) < min_abs_value) {
                new_value = (new_value >= 0 ? min_abs_value : -min_abs_value);
            }
            
            projection(i, j) = new_value;
        }
    }

    // Update bias with similar safeguards
    #pragma omp parallel for
    for (size_t i = 0; i < bias.size(); i++) {
        float update = effective_lr * clip_scale * grad_bias[i];
        
        // Limit bias updates
        const float max_bias_update = 0.05f;
        if (std::abs(update) > max_bias_update) {
            update = (update > 0 ? max_bias_update : -max_bias_update);
        }
        
        bias[i] -= update;
    }
    
    // Print statistics for monitoring
    std::cout << "Backward pass statistics:\n"
              << "Gradient norm: " << grad_norm << "\n"
              << "Max gradient: " << max_grad << "\n"
              << "Effective learning rate: " << effective_lr << "\n"
              << "Clip scale: " << clip_scale << "\n\n";
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
