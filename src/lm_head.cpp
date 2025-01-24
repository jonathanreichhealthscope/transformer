#include "../include/lm_head.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

// Forward declarations of CUDA kernels
__global__ void convert_projection_to_fp16_kernel(
    half* output, const float* input, const unsigned char* active_tokens,
    size_t hidden_size, size_t vocab_size);

LanguageModelHead::LanguageModelHead(size_t hidden_size, size_t vocab_size)
    : hidden_size_(hidden_size), vocab_size_(vocab_size), projection(hidden_size, vocab_size),
      bias(vocab_size, 0.0f), token_frequencies(vocab_size, 0.0f), pruning_threshold(1e-6f),
      active_tokens(vocab_size, 1), training_steps(0)
{
    float scale = std::sqrt(1.0f / hidden_size);
    projection.randomize(-scale, scale);
    bias.randomize(-scale, scale);
    
    // Initialize active token indices with all tokens
    active_token_indices.reserve(vocab_size);
    for (size_t i = 0; i < vocab_size; i++) {
        active_token_indices.push_back(i);
    }
    
#ifdef USE_CUDA
    // Initialize cuBLAS handle
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    
    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    
    // Allocate device memory with maximum sizes
    CUDA_CHECK(cudaMalloc(&d_projection, hidden_size * vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_projection_fp16, hidden_size * vocab_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_hidden_states_fp16, max_batch_size * hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output_fp16, max_batch_size * vocab_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, max_batch_size * vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_active_tokens, vocab_size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_active_token_indices, vocab_size * sizeof(int)));
    
    // Copy initial data to device
    CUDA_CHECK(cudaMemcpyAsync(d_projection, projection.data(), 
                              hidden_size * vocab_size * sizeof(float), 
                              cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_bias, bias.data(), 
                              vocab_size * sizeof(float), 
                              cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_active_tokens, active_tokens.data(),
                              vocab_size * sizeof(unsigned char),
                              cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_active_token_indices, active_token_indices.data(),
                              vocab_size * sizeof(int),
                              cudaMemcpyHostToDevice, compute_stream));
#endif
}

LanguageModelHead::~LanguageModelHead() {
#ifdef USE_CUDA
    if (cublas_handle) cublasDestroy(cublas_handle);
    
    cudaStreamSynchronize(compute_stream);
    cudaStreamDestroy(compute_stream);
    
    if (d_projection) cudaFree(d_projection);
    if (d_bias) cudaFree(d_bias);
    if (d_projection_fp16) cudaFree(d_projection_fp16);
    if (d_hidden_states_fp16) cudaFree(d_hidden_states_fp16);
    if (d_output_fp16) cudaFree(d_output_fp16);
    if (d_output) cudaFree(d_output);
    if (d_active_tokens) cudaFree(d_active_tokens);
    if (d_active_token_indices) cudaFree(d_active_token_indices);
    if (h_projection) cudaFreeHost(h_projection);
    if (h_bias) cudaFreeHost(h_bias);
#endif
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
    
    // Add bias
    for (size_t i = 0; i < logits.rows(); ++i) {
        for (size_t j = 0; j < logits.cols(); ++j) {
            logits(i, j) += bias[j];
        }
    }
    
    return logits;  // Return full logits matrix without reshaping
}

Matrix LanguageModelHead::project_to_vocab(const Matrix& hidden_states) {
    this->hidden_states = hidden_states;
    size_t total_size = hidden_states.rows();  // This is batch_size * seq_len
    size_t hidden_dim = hidden_states.cols();  // This should be hidden_size_
    
    // Calculate actual sequence length from hidden states
    if (hidden_dim != hidden_size_) {
        throw std::runtime_error("Hidden dimension mismatch: " + std::to_string(hidden_dim) +
                               " != " + std::to_string(hidden_size_));
    }
    
    if (total_size > max_batch_size) {
        throw std::runtime_error("Total sequence length exceeds maximum allocated size");
    }
    
    // Update active tokens based on frequency
    if (training_steps % PRUNE_INTERVAL == 0) {
        update_active_tokens();
    }
    
    size_t active_vocab_size = active_token_indices.size();

#ifdef USE_CUDA
    try {
        // Copy input data to device and convert to FP16
        CUDA_CHECK(cudaMemcpyAsync(d_projection, hidden_states.data(), 
                                 total_size * hidden_size_ * sizeof(float),
                                 cudaMemcpyHostToDevice, compute_stream));

        // Convert input to FP16
        launch_convert_to_fp16(d_hidden_states_fp16, d_projection,
                             total_size * hidden_size_);
        
        // Convert projection matrix to FP16
        const int block_size = 256;
        const int num_blocks = (hidden_size_ * vocab_size_ + block_size - 1) / block_size;
        
        convert_projection_to_fp16_kernel<<<num_blocks, block_size, 0, compute_stream>>>(
            d_projection_fp16, d_projection, d_active_tokens,
            hidden_size_, vocab_size_);

        // Use cuBLAS for FP16 matrix multiplication
        const half alpha = __float2half(1.0f);
        const half beta = __float2half(0.0f);
        
        CUBLAS_CHECK(cublasSetStream(cublas_handle, compute_stream));
        CUBLAS_CHECK(cublasGemmEx(cublas_handle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_T,
                                total_size,  // Keep original dimensions
                                active_vocab_size,
                                hidden_size_,
                                &alpha,
                                d_hidden_states_fp16, CUDA_R_16F, hidden_size_,
                                d_projection_fp16, CUDA_R_16F, active_vocab_size,
                                &beta,
                                d_output_fp16, CUDA_R_16F, total_size,
                                CUDA_R_16F,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        
        // Create output matrix with original dimensions
        Matrix logits(total_size, vocab_size_);
        
        // Copy result back to host asynchronously
        CUDA_CHECK(cudaMemcpyAsync(logits.data(), d_output,
                                 total_size * vocab_size_ * sizeof(float),
                                 cudaMemcpyDeviceToHost, compute_stream));
        
        // Make sure all operations are complete
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        
        return logits;  // Return full logits matrix without reshaping
    } catch (const std::runtime_error& e) {
        std::cerr << "CUDA projection failed, falling back to CPU: " << e.what() << std::endl;
        return forward_impl(hidden_states);
    }
#else
    return forward_impl(hidden_states);
#endif
}

Matrix LanguageModelHead::backward(const Matrix& grad_output, const Matrix& target_distribution) {
    size_t total_size = hidden_states.rows();  // This is batch_size * seq_len
    size_t batch_size = total_size / 2;  // Since seq_len is 2
    size_t seq_len = 2;  // Hardcoded for now
    
    std::cout << "LM Head backward dimensions:" << std::endl;
    std::cout << "grad_output: " << grad_output.rows() << "x" << grad_output.cols() << std::endl;
    std::cout << "projection: " << projection.rows() << "x" << projection.cols() << std::endl;
    std::cout << "hidden_states: " << hidden_states.rows() << "x" << hidden_states.cols() << std::endl;

    // Compute cross entropy gradient with respect to logits
    Matrix loss_grad(grad_output.rows(), grad_output.cols());

    if (!target_distribution.empty()) {
            std::cout << "target_distribution: " << target_distribution.rows() << "x"
                    << target_distribution.cols() << std::endl;
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
    expanded_grad.fill(0.0f);  // Initialize with zeros
    
    // Only set gradients for the last token in each sequence
    for (size_t b = 0; b < batch_size; b++) {
        size_t src_idx = b;  // Index in loss_grad
        size_t dst_idx = (b * seq_len + (seq_len - 1));  // Index in expanded_grad (last token position)
        for (size_t v = 0; v < vocab_size_; v++) {
            expanded_grad(dst_idx, v) = loss_grad(src_idx, v);
        }
    }

    // Propagate gradients through the linear layer with expanded gradients
    backward_linear(expanded_grad);

    // Return gradients with respect to hidden states
    Matrix hidden_grad = matmul(expanded_grad, projection.transpose());
    
    return hidden_grad;  // No need to reshape since dimensions already match hidden_states
}

void LanguageModelHead::backward_linear(const Matrix& grad_output) {
    // Check dimensions before matrix multiplication
    if (grad_output.rows() != hidden_states.rows()) {
        throw std::runtime_error("Invalid matrix dimensions for gradient computation: " +
                                 std::to_string(grad_output.rows()) +
                                 " != " + std::to_string(hidden_states.rows()));
    }

    // Compute gradients for projection matrix
    Matrix grad_proj = matmul(hidden_states.transpose(), grad_output);

    // Verify gradient dimensions
    if (grad_proj.rows() != projection.rows() || grad_proj.cols() != projection.cols()) {
        throw std::runtime_error(
            "Gradient dimensions don't match projection matrix: " +
            std::to_string(grad_proj.rows()) + "x" + std::to_string(grad_proj.cols()) + " vs " +
            std::to_string(projection.rows()) + "x" + std::to_string(projection.cols()));
    }

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
    
#ifdef USE_CUDA
    // Update device active tokens and indices
    CUDA_CHECK(cudaMemcpyAsync(d_active_tokens, active_tokens.data(),
                              vocab_size_ * sizeof(unsigned char),
                              cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_active_token_indices, active_token_indices.data(),
                              active_token_indices.size() * sizeof(int),
                              cudaMemcpyHostToDevice, compute_stream));
#endif
}

void LanguageModelHead::prune_vocabulary(float min_frequency_threshold) {
    // Count total occurrences
    float total_occurrences = std::accumulate(token_frequencies.begin(), 
                                            token_frequencies.end(), 0.0f);
    
    if (total_occurrences < 1e-10f) {
        std::cerr << "Warning: No token occurrences recorded, skipping pruning" << std::endl;
        return;
    }
    
    // Create mapping for tokens to keep
    std::vector<int> keep_tokens;
    std::vector<int> token_mapping(vocab_size_, -1);
    
    // Keep special tokens (assuming first few tokens are special)
    for (int i = 0; i < 5; i++) {  // Adjust based on your special token count
        keep_tokens.push_back(i);
        token_mapping[i] = i;
    }
    
    // Keep frequent tokens
    for (size_t i = 5; i < vocab_size_; i++) {
        float frequency = token_frequencies[i] / total_occurrences;
        if (frequency >= min_frequency_threshold) {
            token_mapping[i] = keep_tokens.size();
            keep_tokens.push_back(i);
        }
    }
    
    // Ensure we keep at least MIN_ACTIVE_TOKENS
    if (keep_tokens.size() < MIN_ACTIVE_TOKENS) {
        std::vector<std::pair<float, size_t>> freq_pairs;
        freq_pairs.reserve(vocab_size_);
        for (size_t i = 0; i < vocab_size_; i++) {
            freq_pairs.push_back({token_frequencies[i], i});
        }
        
        std::partial_sort(freq_pairs.begin(), 
                         freq_pairs.begin() + MIN_ACTIVE_TOKENS,
                         freq_pairs.end(),
                         std::greater<>());
        
        keep_tokens.clear();
        token_mapping.assign(vocab_size_, -1);
        for (size_t i = 0; i < MIN_ACTIVE_TOKENS; i++) {
            size_t idx = freq_pairs[i].second;
            token_mapping[idx] = i;
            keep_tokens.push_back(idx);
        }
    }
    
    // Resize projection matrix and bias
    size_t new_vocab_size = keep_tokens.size();
    Matrix new_projection(hidden_size_, new_vocab_size);
    Vector new_bias(new_vocab_size);
    std::vector<float> new_frequencies(new_vocab_size);
    
    // Copy kept tokens' weights and frequencies
    for (size_t i = 0; i < keep_tokens.size(); i++) {
        int old_idx = keep_tokens[i];
        for (size_t j = 0; j < hidden_size_; j++) {
            new_projection(j, i) = projection(j, old_idx);
        }
        new_bias[i] = bias[old_idx];
        new_frequencies[i] = token_frequencies[old_idx];
    }
    
    // Update class members
    projection = std::move(new_projection);
    bias = std::move(new_bias);
    token_frequencies = std::move(new_frequencies);
    vocab_size_ = new_vocab_size;
    
    // Update active tokens
    active_tokens.resize(vocab_size_);
    std::fill(active_tokens.begin(), active_tokens.end(), 1);
    
    active_token_indices.clear();
    active_token_indices.reserve(vocab_size_);
    for (size_t i = 0; i < vocab_size_; i++) {
        active_token_indices.push_back(i);
    }
    
#ifdef USE_CUDA
    // Update device memory
    CUDA_CHECK(cudaMemcpyAsync(d_projection, projection.data(),
                              hidden_size_ * vocab_size_ * sizeof(float),
                              cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_bias, bias.data(),
                              vocab_size_ * sizeof(float),
                              cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_active_tokens, active_tokens.data(),
                              vocab_size_ * sizeof(unsigned char),
                              cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_active_token_indices, active_token_indices.data(),
                              vocab_size_ * sizeof(int),
                              cudaMemcpyHostToDevice, compute_stream));
#endif

    std::cout << "Pruned vocabulary from " << token_mapping.size() 
              << " to " << new_vocab_size << " tokens\n";
}

#ifdef USE_CUDA
// Add these CUDA kernel definitions and launchers

// Kernel for converting FP32 to FP16
__global__ void convert_to_fp16_kernel(half* output, const float* input, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

// Kernel for converting and expanding vocabulary
__global__ void convert_and_expand_vocab_kernel(
    float* output, const half* input, const unsigned char* active_tokens,
    size_t batch_size, size_t vocab_size, size_t active_vocab_size) {
    
    const size_t row = blockIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < vocab_size) {
        // Count active tokens before this position to determine input index
        size_t active_idx = 0;
        for (size_t i = 0; i < col && active_idx < active_vocab_size; i++) {
            if (active_tokens[i]) active_idx++;
        }
        
        if (active_tokens[col] && active_idx < active_vocab_size) {
            output[row * vocab_size + col] = __half2float(input[row * active_vocab_size + active_idx]);
        } else {
            output[row * vocab_size + col] = -INFINITY;
        }
    }
}

namespace {
    // Device function declarations
    __device__ void convert_to_fp16_device(half* output, const float* input, size_t idx) {
        output[idx] = __float2half(input[idx]);
    }

    __device__ void convert_and_expand_vocab_device(
        float* output, const half* input, const unsigned char* active_tokens,
        size_t row, size_t col, size_t batch_size, size_t vocab_size, size_t active_vocab_size) {
        
        if (row < batch_size && col < vocab_size) {
            size_t active_idx = 0;
            for (size_t i = 0; i < col && active_idx < active_vocab_size; i++) {
                if (active_tokens[i]) active_idx++;
            }
            
            if (active_tokens[col] && active_idx < active_vocab_size) {
                output[row * vocab_size + col] = __half2float(input[row * active_vocab_size + active_idx]);
            } else {
                output[row * vocab_size + col] = -INFINITY;
            }
        }
    }
}

// Launcher for FP32 to FP16 conversion
__host__ void LanguageModelHead::launch_convert_to_fp16(half* output, const float* input, size_t size) {
    // First check if pointers are valid
    if (output == nullptr || input == nullptr) {
        throw std::runtime_error("Null pointer passed to launch_convert_to_fp16");
    }

    // Verify these are device pointers
    cudaPointerAttributes output_attr, input_attr;
    CUDA_CHECK(cudaPointerGetAttributes(&output_attr, output));
    CUDA_CHECK(cudaPointerGetAttributes(&input_attr, input));
    
    if (output_attr.type != cudaMemoryTypeDevice || input_attr.type != cudaMemoryTypeDevice) {
        throw std::runtime_error("Non-device memory pointer passed to launch_convert_to_fp16");
    }

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    convert_to_fp16_kernel<<<num_blocks, block_size, 0, nullptr>>>(output, input, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Launcher for vocabulary expansion
__host__ void LanguageModelHead::launch_convert_and_expand_vocab(
    float* output, const half* input, size_t batch_size, size_t vocab_size, size_t active_vocab_size) {
    
    const int block_size = 256;
    const int num_blocks_x = (vocab_size + block_size - 1) / block_size;
    
    dim3 grid(num_blocks_x, batch_size);
    dim3 block(block_size);
    
    convert_and_expand_vocab_kernel<<<grid, block, 0, nullptr>>>(
        output, input, d_active_tokens,
        batch_size, vocab_size, active_vocab_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

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