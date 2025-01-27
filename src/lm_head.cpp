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
    float scale = std::sqrt(6.0f / (hidden_size + vocab_size));  // Proper Xavier scaling
    projection.randomize(-scale, scale);
    
    // Initialize bias to zero instead of negative values
    #pragma omp parallel for
    for (size_t i = 0; i < vocab_size; i++) {
        bias[i] = 0.0f;
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
    
    // Debug projection matrix using sampling
    float min_proj = std::numeric_limits<float>::infinity();
    float max_proj = -std::numeric_limits<float>::infinity();
    float sum_proj = 0.0f;
    size_t nonzero_proj = 0;
    
    // Only compute expensive statistics in debug mode or first forward pass
    static bool first_pass = true;
    if (first_pass || !is_training_) {
        const size_t total_elements = projection.rows() * projection.cols();
        // Sample ~1% of elements or 10000, whichever is smaller
        const size_t sample_size = std::min(total_elements / 100, size_t(10000));
        const float* data = projection.data();
        
        // Use thread-safe random number generation
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dist(0, total_elements - 1);
        
        #pragma omp parallel
        {
            float local_min = std::numeric_limits<float>::infinity();
            float local_max = -std::numeric_limits<float>::infinity();
            float local_sum = 0.0f;
            size_t local_nonzero = 0;
            size_t local_samples = 0;
            
            // Each thread processes its share of samples
            #pragma omp for
            for (size_t i = 0; i < sample_size; i++) {
                size_t idx = dist(gen);
                float val = data[idx];
                local_min = std::min(local_min, val);
                local_max = std::max(local_max, val);
                local_sum += val;
                if (std::abs(val) > 1e-6) local_nonzero++;
                local_samples++;
            }
            
            // Scale up the counts to estimate full matrix
            float scale_factor = float(total_elements) / sample_size;
            local_sum *= scale_factor;
            local_nonzero = size_t(local_nonzero * scale_factor);
            
            #pragma omp critical
            {
                min_proj = std::min(min_proj, local_min);
                max_proj = std::max(max_proj, local_max);
                sum_proj += local_sum;
                nonzero_proj += local_nonzero;
            }
        }
        first_pass = false;
    }
    
    std::cout << "Projection Matrix Statistics (Estimated from sampling):\n"
              << "Min proj: " << min_proj << "\n"
              << "Max proj: " << max_proj << "\n"
              << "Mean proj: " << sum_proj / (projection.rows() * projection.cols()) << "\n"
              << "Nonzero proj: " << nonzero_proj << "/" 
              << (projection.rows() * projection.cols()) << "\n\n";
    
    // Compute logits with proper scaling
    Matrix logits = matmul(scaled_hidden, projection);
    
    // Add bias and apply proper scaling
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < logits.rows(); ++i) {
        for (size_t j = 0; j < logits.cols(); ++j) {
            // Add bias
            logits(i, j) += bias[j];
            
            // Apply token-specific adjustments
            if (j == static_cast<size_t>(tokens::UNK_ID)) {
                logits(i, j) -= 2.0f;  // Reduced penalty for UNK
            } else if (token_frequencies[j] < 1e-6) {
                logits(i, j) -= 1.0f;  // Smaller penalty for rare tokens
            }
        }
    }
    
    // Apply layer normalization to logits for better numerical stability
    #pragma omp parallel for
    for (size_t i = 0; i < logits.rows(); ++i) {
        float row_mean = 0.0f;
        float row_var = 0.0f;
        
        // Compute mean
        for (size_t j = 0; j < logits.cols(); ++j) {
            row_mean += logits(i, j);
        }
        row_mean /= logits.cols();
        
        // Compute variance
        for (size_t j = 0; j < logits.cols(); ++j) {
            float diff = logits(i, j) - row_mean;
            row_var += diff * diff;
        }
        row_var = std::sqrt(row_var / logits.cols() + 1e-5f);
        
        // Normalize and scale
        for (size_t j = 0; j < logits.cols(); ++j) {
            logits(i, j) = (logits(i, j) - row_mean) / row_var;
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
    float scale_factor = std::sqrt(2.0f / hidden_size_);  // Match forward pass scaling
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < scaled_hidden.rows(); i++) {
        for (size_t j = 0; j < scaled_hidden.cols(); j++) {
            scaled_hidden(i, j) *= scale_factor;
        }
    }

    Matrix grad_proj = matmul(scaled_hidden.transpose(), grad_output);
    Vector grad_bias(bias.size(), 0.0f);

    // Compute bias gradients without array section syntax
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

    // Use adaptive learning rate based on gradient magnitude
    float grad_norm = 0.0f;
    #pragma omp parallel for reduction(+:grad_norm)
    for (size_t i = 0; i < grad_proj.rows(); i++) {
        for (size_t j = 0; j < grad_proj.cols(); j++) {
            grad_norm += grad_proj(i, j) * grad_proj(i, j);
        }
    }
    grad_norm = std::sqrt(grad_norm);
    
    // Clip learning rate based on gradient norm
    const float base_lr = 0.01f;
    float effective_lr = std::min(base_lr, base_lr / (1.0f + grad_norm));
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < projection.rows(); i++) {
        for (size_t j = 0; j < projection.cols(); j++) {
            projection(i, j) -= effective_lr * grad_proj(i, j);
        }
    }

    #pragma omp parallel for
    for (size_t i = 0; i < bias.size(); i++) {
        bias[i] -= effective_lr * grad_bias[i];
    }
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
