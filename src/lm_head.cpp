#include "../include/lm_head.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

LanguageModelHead::LanguageModelHead(size_t hidden_size, size_t vocab_size)
    : hidden_size_(hidden_size), vocab_size_(vocab_size), projection(hidden_size, vocab_size),
      bias(vocab_size, 0.0f), token_frequencies(vocab_size, 0.0f), pruning_threshold(1e-6f),
      active_tokens(vocab_size, 1), training_steps(0)
{
    float scale = std::sqrt(2.0f / (hidden_size + vocab_size));  // Xavier initialization
    projection.randomize(-scale, scale);
    bias.randomize(-scale, scale);
    
    // Initialize active token indices with all tokens
    active_token_indices.reserve(vocab_size);
    for (size_t i = 0; i < vocab_size; i++) {
        active_token_indices.push_back(i);
    }
}

Matrix LanguageModelHead::forward_impl(const Matrix& hidden_states) {
    size_t total_size = hidden_states.rows();
    size_t hidden_dim = hidden_states.cols();
    
    if (hidden_dim != hidden_size_) {
        throw std::runtime_error("Hidden dimension mismatch: " + std::to_string(hidden_dim) +
                               " != " + std::to_string(hidden_size_));
    }
    
    // Compute logits using the original dimensions
    Matrix logits = matmul(hidden_states, projection);
    
    // Add bias with parallel processing
    #pragma omp parallel for collapse(2)
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
    
    return forward_impl(hidden_states);
}

Matrix LanguageModelHead::backward(const Matrix& grad_output, const Matrix& target_distribution) {
    size_t total_size = hidden_states.rows();
    size_t batch_size = total_size / 2;
    size_t seq_len = 2;
    
    // Compute cross entropy gradient with respect to logits
    Matrix loss_grad(grad_output.rows(), grad_output.cols());

    if (!target_distribution.empty()) {
        // If target distribution is provided, compute cross entropy gradient
        for (size_t i = 0; i < grad_output.rows(); i++) {
            for (size_t j = 0; j < grad_output.cols(); j++) {
                if (target_distribution(i, j) > 0.0f) {
                    loss_grad(i, j) = grad_output(i, j) - target_distribution(i, j);
                }
            }
        }
    } else {
        // Otherwise, just use the provided gradients
        loss_grad = grad_output;
    }

    // Expand the gradients back to full sequence length
    Matrix expanded_grad(total_size, vocab_size_);
    expanded_grad.fill(0.0f);
    
    // Only set gradients for the last token in each sequence
    for (size_t b = 0; b < batch_size; b++) {
        size_t src_idx = b;
        size_t dst_idx = (b * seq_len + (seq_len - 1));
        for (size_t v = 0; v < vocab_size_; v++) {
            expanded_grad(dst_idx, v) = loss_grad(src_idx, v);
        }
    }

    // Propagate gradients through the linear layer with expanded gradients
    backward_linear(expanded_grad);

    // Return gradients with respect to hidden states
    return matmul(expanded_grad, projection.transpose());
}

void LanguageModelHead::backward_linear(const Matrix& grad_output) {
    if (grad_output.rows() != hidden_states.rows()) {
        throw std::runtime_error("Invalid matrix dimensions for gradient computation");
    }

    // Compute gradients for projection matrix
    Matrix grad_proj = matmul(hidden_states.transpose(), grad_output);

    // Compute gradients for bias
    Vector grad_bias(bias.size(), 0.0f);
    for (size_t i = 0; i < grad_output.rows(); i++) {
        for (size_t j = 0; j < grad_output.cols(); j++) {
            grad_bias[j] += grad_output(i, j);
        }
    }

    // Update parameters using gradients
    const float learning_rate = 0.001f;
    
    // Update projection matrix
    for (size_t i = 0; i < projection.rows(); i++) {
        for (size_t j = 0; j < projection.cols(); j++) {
            projection(i, j) -= learning_rate * grad_proj(i, j);
        }
    }

    // Update bias
    for (size_t i = 0; i < bias.size(); i++) {
        bias[i] -= learning_rate * grad_bias[i];
    }
}

void LanguageModelHead::update_active_tokens() {
    // Update token frequencies with exponential decay
    const float decay = 0.99f;
    for (size_t i = 0; i < vocab_size_; i++) {
        token_frequencies[i] *= decay;
    }
    
    // Count tokens above threshold
    size_t active_count = 0;
    active_token_indices.clear();
    
    for (size_t i = 0; i < vocab_size_; i++) {
        active_tokens[i] = (token_frequencies[i] > pruning_threshold) ? 1 : 0;
        if (active_tokens[i]) {
            active_token_indices.push_back(i);
            active_count++;
        }
    }
    
    // Ensure we keep at least MIN_ACTIVE_TOKENS
    if (active_count < MIN_ACTIVE_TOKENS) {
        std::vector<std::pair<float, size_t>> freq_pairs;
        freq_pairs.reserve(vocab_size_);
        for (size_t i = 0; i < vocab_size_; i++) {
            freq_pairs.push_back({token_frequencies[i], i});
        }
        
        std::partial_sort(freq_pairs.begin(), 
                         freq_pairs.begin() + MIN_ACTIVE_TOKENS,
                         freq_pairs.end(),
                         std::greater<>());
        
        active_token_indices.clear();
        std::fill(active_tokens.begin(), active_tokens.end(), 0);
        for (size_t i = 0; i < MIN_ACTIVE_TOKENS; i++) {
            size_t idx = freq_pairs[i].second;
            active_tokens[idx] = 1;
            active_token_indices.push_back(idx);
        }
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