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
      bias(vocab_size, 0.0f), token_frequencies(vocab_size, 0.0f)
{
    float scale = std::sqrt(1.0f / hidden_size);
    projection.randomize(-scale, scale);
    bias.randomize(-scale, scale);
    
    // Initialize pruning threshold and active token mask
    pruning_threshold = 1e-6f;
    active_tokens.resize(vocab_size, 1);  // 1 = true
    
#ifdef USE_CUDA
    // Initialize cuBLAS
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_projection, hidden_size * vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_projection_fp16, hidden_size * vocab_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_active_tokens, vocab_size * sizeof(unsigned char)));
    
    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_projection, projection.data(), 
                         hidden_size * vocab_size * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bias.data(), 
                         vocab_size * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_active_tokens, 1, vocab_size * sizeof(unsigned char)));
#endif
}

LanguageModelHead::~LanguageModelHead() {
#ifdef USE_CUDA
    // Destroy cuBLAS handle
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
    if (h_projection) cudaFreeHost(h_projection);
    if (h_bias) cudaFreeHost(h_bias);
#endif
}

Matrix LanguageModelHead::forward_impl(const Matrix& hidden_states) {
    // Store hidden states for backward pass
    this->hidden_states = hidden_states;

    std::cout << "In language Model Head: " << std::endl;
    std::cout << "Hidden states dimensions: " << hidden_states.rows() << "x" << hidden_states.cols()
              << std::endl;
    std::cout << "Projection dimensions: " << projection.rows() << "x" << projection.cols()
              << std::endl;

    // Check dimensions before multiplication
    if (hidden_states.cols() != projection.rows()) {
        throw std::runtime_error(
            "Invalid matrix dimensions for projection: hidden_states.cols() (" +
            std::to_string(hidden_states.cols()) + ") must match projection.rows() (" +
            std::to_string(projection.rows()) + ")");
    }

    // Project hidden states to vocabulary size
    // [batch_size x hidden_size] * [hidden_size x vocab_size] = [batch_size x vocab_size]
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
    size_t batch_size = hidden_states.rows();
    
    // Update active tokens based on frequency
    if (training_steps % PRUNE_INTERVAL == 0) {
        update_active_tokens();
    }
    
    // Count active tokens
    size_t active_vocab_size = std::count(active_tokens.begin(), active_tokens.end(), 1);

#ifdef USE_CUDA
    try {
        // Allocate all device memory first
        if (d_hidden_states_fp16 == nullptr) {
            CUDA_CHECK(cudaMalloc(&d_hidden_states_fp16, batch_size * hidden_size_ * sizeof(half)));
        }
        if (d_output_fp16 == nullptr) {
            CUDA_CHECK(cudaMalloc(&d_output_fp16, batch_size * active_vocab_size * sizeof(half)));
        }
        if (d_output == nullptr) {
            CUDA_CHECK(cudaMalloc(&d_output, batch_size * vocab_size_ * sizeof(float)));
        }

        // Allocate and copy input data
        float* d_input = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input, batch_size * hidden_size_ * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, hidden_states.data(), 
                            batch_size * hidden_size_ * sizeof(float),
                            cudaMemcpyHostToDevice));

        // Now we can safely convert to FP16
        launch_convert_to_fp16(d_hidden_states_fp16, d_input,
                             batch_size * hidden_size_);

        // Free temporary input buffer
        CUDA_CHECK(cudaFree(d_input));

        // Use FP16 for projection matrix to reduce memory and increase speed
        half *d_projection_fp16, *d_hidden_states_fp16, *d_output_fp16;
        float *d_output;
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_projection_fp16, hidden_size_ * active_vocab_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_hidden_states_fp16, batch_size * hidden_size_ * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_output_fp16, batch_size * active_vocab_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_output, batch_size * vocab_size_ * sizeof(float)));
        
        // Convert and copy active tokens to FP16
        std::vector<half> h_projection_fp16;
        h_projection_fp16.reserve(hidden_size_ * active_vocab_size);
        for (size_t i = 0; i < vocab_size_; i++) {
            if (active_tokens[i]) {
                for (size_t j = 0; j < hidden_size_; j++) {
                    h_projection_fp16.push_back(__float2half(projection(j, i)));
                }
            }
        }
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpyAsync(d_projection_fp16, h_projection_fp16.data(),
                                  hidden_size_ * active_vocab_size * sizeof(half),
                                  cudaMemcpyHostToDevice));
                                  
        // Convert input to FP16
        launch_convert_to_fp16(d_hidden_states_fp16, d_input,
                             batch_size * hidden_size_);
        
        // Use cuBLAS for FP16 matrix multiplication
        const half alpha = __float2half(1.0f);
        const half beta = __float2half(0.0f);
        
        // Note: cuBLAS uses column-major order, so we need to transpose our operation
        // M = active_vocab_size (output rows)
        // N = batch_size (output cols)
        // K = hidden_size (inner dimension)
        CUBLAS_CHECK(cublasGemmEx(cublas_handle,
                                CUBLAS_OP_T,  // Transpose first matrix
                                CUBLAS_OP_T,  // Transpose second matrix
                                batch_size,    // M: number of rows of output
                                active_vocab_size, // N: number of columns of output
                                hidden_size_,  // K: inner dimension
                                &alpha,
                                d_hidden_states_fp16, CUDA_R_16F, hidden_size_,  // lda = K
                                d_projection_fp16, CUDA_R_16F, active_vocab_size,  // ldb = N
                                &beta,
                                d_output_fp16, CUDA_R_16F, batch_size,  // ldc = M
                                CUDA_R_16F,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        
        // Convert back to FP32 and expand to full vocabulary
        launch_convert_and_expand_vocab(d_output, d_output_fp16,
                                      batch_size, vocab_size_, active_vocab_size);
        
        // Create output matrix
        Matrix logits(batch_size, vocab_size_);
        
        // Copy result back to host
        CUDA_CHECK(cudaMemcpyAsync(logits.data(), d_output,
                                  batch_size * vocab_size_ * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        
        // Add bias only for active tokens
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < vocab_size_; ++j) {
                if (active_tokens[j]) {
                    logits(i, j) += bias[j];
                } else {
                    logits(i, j) = -std::numeric_limits<float>::infinity();
                }
            }
        }
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_projection_fp16));
        CUDA_CHECK(cudaFree(d_hidden_states_fp16));
        CUDA_CHECK(cudaFree(d_output_fp16));
        CUDA_CHECK(cudaFree(d_output));
        
        return logits;
    } catch (const std::runtime_error& e) {
        std::cerr << "CUDA projection failed, falling back to CPU: " << e.what() << std::endl;
#endif
        // CPU fallback with sparse computation
        Matrix logits(batch_size, vocab_size_);
        
        // Use OpenMP for parallel processing
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < vocab_size_; ++j) {
                if (!active_tokens[j]) {
                    logits(i, j) = -std::numeric_limits<float>::infinity();
                    continue;
                }
                float sum = 0.0f;
                // Manual loop unrolling for better CPU optimization
                for (size_t k = 0; k < hidden_size_; k += 4) {
                    sum += hidden_states(i, k) * projection(k, j);
                    if (k + 1 < hidden_size_)
                        sum += hidden_states(i, k+1) * projection(k+1, j);
                    if (k + 2 < hidden_size_)
                        sum += hidden_states(i, k+2) * projection(k+2, j);
                    if (k + 3 < hidden_size_)
                        sum += hidden_states(i, k+3) * projection(k+3, j);
                }
                logits(i, j) = sum + bias[j];
            }
        }
        return logits;
#ifdef USE_CUDA
    }
#endif
}

Matrix LanguageModelHead::backward(const Matrix& grad_output, const Matrix& target_distribution) {
    std::cout << "LM Head backward dimensions:" << std::endl;
    std::cout << "grad_output: " << grad_output.rows() << "x" << grad_output.cols() << std::endl;
    std::cout << "projection: " << projection.rows() << "x" << projection.cols() << std::endl;
    std::cout << "hidden_states: " << hidden_states.rows() << "x" << hidden_states.cols()
              << std::endl;

    // Compute cross entropy gradient with respect to logits
    Matrix loss_grad(grad_output.rows(), grad_output.cols());

    if (!target_distribution.empty()) {
        std::cout << "target_distribution: " << target_distribution.rows() << "x"
                  << target_distribution.cols() << std::endl;
        // If target distribution is provided, compute cross entropy gradient
        for (size_t i = 0; i < grad_output.rows(); i++) {
            for (size_t j = 0; j < grad_output.cols(); j++) {
                size_t idx = i * grad_output.cols() + j;
                if (target_distribution.data()[i] > 0.0f) {
                    loss_grad(i, j) = grad_output(i, j) - target_distribution(i, j);
                }
            }
        }
    } else {
        // Otherwise, just use the provided gradients
        loss_grad = grad_output;
    }

    // Propagate gradients through the linear layer
    backward_linear(loss_grad);

    // Return gradients with respect to hidden states
    // loss_grad: [batch_size x vocab_size], projection: [hidden_size x vocab_size]
    // Need to transpose projection to get [vocab_size x hidden_size]
    return matmul(loss_grad, projection.transpose());
}

void LanguageModelHead::backward_linear(const Matrix& grad_output) {
    // Check dimensions before matrix multiplication
    if (grad_output.rows() != hidden_states.rows()) {
        throw std::runtime_error("Invalid matrix dimensions for gradient computation: " +
                                 std::to_string(grad_output.rows()) +
                                 " != " + std::to_string(hidden_states.rows()));
    }

    // Compute gradients for projection matrix
    // hidden_states: [batch_size x hidden_size], grad_output: [batch_size x vocab_size]
    // Result should be [hidden_size x vocab_size]
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
    const float learning_rate = 0.001f; // You might want to make this configurable

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
    for (size_t i = 0; i < vocab_size_; i++) {
        active_tokens[i] = (token_frequencies[i] > pruning_threshold) ? 1 : 0;
        if (active_tokens[i]) active_count++;
    }
    
    // Ensure we keep at least MIN_ACTIVE_TOKENS
    if (active_count < MIN_ACTIVE_TOKENS) {
        std::vector<std::pair<float, size_t>> freq_pairs;
        freq_pairs.reserve(vocab_size_);
        for (size_t i = 0; i < vocab_size_; i++) {
            freq_pairs.push_back({token_frequencies[i], i});
        }
        
        // Sort by frequency
        std::partial_sort(freq_pairs.begin(), 
                         freq_pairs.begin() + MIN_ACTIVE_TOKENS,
                         freq_pairs.end(),
                         std::greater<>());
                         
        // Update active tokens
        std::fill(active_tokens.begin(), active_tokens.end(), 0);
        for (size_t i = 0; i < MIN_ACTIVE_TOKENS; i++) {
            active_tokens[freq_pairs[i].second] = 1;
        }
    }
    
#ifdef USE_CUDA
    // Update device active tokens
    CUDA_CHECK(cudaMemcpy(d_active_tokens, active_tokens.data(),
                         vocab_size_ * sizeof(unsigned char),
                         cudaMemcpyHostToDevice));
#endif
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
#endif