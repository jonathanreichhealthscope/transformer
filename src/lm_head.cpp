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

// Only include CUDA headers if CUDA is available
#if defined(USE_CUDA) && defined(CUDA_AVAILABLE)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

LanguageModelHead::LanguageModelHead(size_t hidden_size, size_t vocab_size)
    : hidden_size_(hidden_size), vocab_size_(vocab_size), 
      projection(hidden_size, vocab_size),  // [hidden_size x vocab_size] for correct dimensionality
      bias(vocab_size, 0.0f),  // [vocab_size] stays the same
      token_frequencies(vocab_size, 0.0f),
      pruning_threshold(1e-6f),
      active_tokens(vocab_size, 1),
      training_steps(0),
      is_training_(false),
      m_proj(hidden_size, vocab_size, 0.0f),  // Match projection dimensions
      v_proj(hidden_size, vocab_size, 0.0f),  // Match projection dimensions
      m_bias(vocab_size, 0.0f),
      v_bias(vocab_size, 0.0f),
      t(0),
      beta1(0.9f),
      beta2(0.999f),
      eps(1e-8f),
      current_lr(0.001f),
      min_lr(0.0001f),
      max_lr(0.01f),
      lr_decay(0.99f) {
    
    // Initialize projection matrix with Xavier/Glorot initialization
    float scale = std::sqrt(2.0f / (hidden_size + vocab_size));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);
    
    // Initialize projection weights
    for (size_t i = 0; i < projection.rows(); i++) {
        for (size_t j = 0; j < projection.cols(); j++) {
            projection(i, j) = dist(gen);
        }
    }
    
    std::cout << "Initialized LM Head with:" << std::endl;
    std::cout << "- Hidden size: " << hidden_size << std::endl;
    std::cout << "- Vocab size: " << vocab_size << std::endl;
    std::cout << "- Projection matrix: " << projection.rows() << "x" << projection.cols() << std::endl;
    std::cout << "- Projection matrix shape: [hidden_size x vocab_size] = [" << hidden_size << " x " << vocab_size << "]" << std::endl;
}

Matrix LanguageModelHead::forward(const Matrix& hidden_states, bool training) {
    if (hidden_states.cols() != hidden_size_) {
        throw std::runtime_error("Hidden dimension mismatch: " + std::to_string(hidden_states.cols()) +
                               " != " + std::to_string(hidden_size_));
    }
    
    // Cache hidden states for backward pass
    hidden_states_ = hidden_states;
    
    // Project hidden states to vocabulary space using transposed projection
    // hidden_states: [batch_size x hidden_size]
    // projection: [hidden_size x vocab_size]
    // result: [batch_size x vocab_size]
    Matrix logits = matmul(hidden_states, projection);
    
    // Add bias
    for (size_t i = 0; i < logits.rows(); ++i) {
        for (size_t j = 0; j < logits.cols(); ++j) {
            logits(i, j) += bias[j];
        }
    }
    
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
    
    return forward(hidden_states, false);
}

Matrix LanguageModelHead::backward(const Matrix& grad_output, const Matrix& target_distribution) {
    return backward_pass(grad_output, hidden_states);  // Use the existing backward_pass implementation
}

void LanguageModelHead::backward_linear(const Matrix& grad_output) {
    // Use backward_pass which already has Adam optimization
    backward_pass(grad_output, hidden_states_);
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
    // Verify input dimensions
    if (grad_output.cols() != vocab_size_) {
        throw std::runtime_error("Gradient output dimension mismatch in backward pass. Expected vocab_size: " + 
                               std::to_string(vocab_size_) + ", got: " + std::to_string(grad_output.cols()));
    }
    if (hidden_states.cols() != hidden_size_) {
        throw std::runtime_error("Hidden states dimension mismatch in backward pass. Expected hidden_size: " + 
                               std::to_string(hidden_size_) + ", got: " + std::to_string(hidden_states.cols()));
    }

    std::cout << "\nLM Head Backward Pass Dimensions:" << std::endl;
    std::cout << "- grad_output: [" << grad_output.rows() << " x " << grad_output.cols() << "]" << std::endl;
    std::cout << "- hidden_states: [" << hidden_states.rows() << " x " << hidden_states.cols() << "]" << std::endl;
    std::cout << "- projection: [" << projection.rows() << " x " << projection.cols() << "]" << std::endl;
    
    // Compute gradients for projection matrix
    // hidden_states.T: [hidden_size x batch_size]
    // grad_output: [batch_size x vocab_size]
    // grad_proj: [hidden_size x vocab_size]
    Matrix hidden_states_t = hidden_states.transpose();
    Matrix grad_proj = matmul(hidden_states_t, grad_output);
    
    // Compute bias gradients
    Vector grad_bias = grad_output.row_sum();
    
    // Update parameters using Adam optimizer
    t++;  // Increment time step
    
    // Constants for gradient clipping and stability
    const float clip_threshold = 5.0f;
    const float max_allowed_value = 100.0f;
    const float scale_factor = std::sqrt(1.0f / hidden_size_);
    const float max_update = 0.05f * scale_factor;
    
    bool has_unstable_update = false;
    
    // Update projection matrix using Adam optimizer
    #pragma omp parallel for collapse(2) reduction(|:has_unstable_update)
    for (size_t i = 0; i < grad_proj.rows(); ++i) {
        for (size_t j = 0; j < grad_proj.cols(); ++j) {
            if (!std::isfinite(grad_proj(i, j))) {
                continue;
            }
            
            // Clip gradient
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
    
    // Compute gradient with respect to input
    // grad_output: [batch_size x vocab_size]
    // projection.T: [vocab_size x hidden_size]
    // result: [batch_size x hidden_size]
    Matrix projection_t = projection.transpose();
    Matrix grad_input = matmul(grad_output, projection_t);
    
    // Verify output dimensions
    if (grad_input.cols() != hidden_size_) {
        throw std::runtime_error("Output gradient dimension mismatch. Expected hidden_size: " + 
                               std::to_string(hidden_size_) + ", got: " + std::to_string(grad_input.cols()));
    }
    
    std::cout << "Gradient propagation dimensions:" << std::endl;
    std::cout << "- Input gradient: [" << grad_input.rows() << " x " << grad_input.cols() << "]" << std::endl;
    
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
