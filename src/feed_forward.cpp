#include "../include/feed_forward.hpp"
#ifdef USE_CUDA
#include "../include/cuda/cuda_check.cuh"
#include "../include/cuda/cuda_launch.cuh"
#include "../include/cuda/feed_forward_kernels.cuh"
#include "../include/cuda/backward_ops.cuh"
#include "../include/cuda/matrix_ops.cuh"
#include "../include/cuda/memory_manager.cuh"
#endif
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

FeedForward::FeedForward(size_t hidden_size, size_t intermediate_size, float dropout)
    : w1(hidden_size, intermediate_size), w2(intermediate_size, hidden_size), b1(intermediate_size),
      b2(hidden_size), dropout_prob(dropout), intermediate_cache(1, intermediate_size),
      // Initialize gradients with same dimensions as their parameters
      dW1_(hidden_size, intermediate_size), dW2_(intermediate_size, hidden_size),
      db1_(intermediate_size), db2_(hidden_size) {

    // Initialize weights with Xavier/Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());

    float w1_limit = std::sqrt(6.0f / (hidden_size + intermediate_size));
    float w2_limit = std::sqrt(6.0f / (intermediate_size + hidden_size));

    std::uniform_real_distribution<float> w1_dis(-w1_limit, w1_limit);
    std::uniform_real_distribution<float> w2_dis(-w2_limit, w2_limit);

    // Initialize weights
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < w1.rows(); ++i) {
        for (size_t j = 0; j < w1.cols(); ++j) {
            w1(i, j) = w1_dis(gen);
        }
    }

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < w2.rows(); ++i) {
        for (size_t j = 0; j < w2.cols(); ++j) {
            w2(i, j) = w2_dis(gen);
        }
    }

    // Initialize biases to zero
    #pragma omp parallel for
    for (size_t i = 0; i < b1.size(); ++i)
        b1[i] = 0.0f;
    #pragma omp parallel for
    for (size_t i = 0; i < b2.size(); ++i)
        b2[i] = 0.0f;

    // Initialize gradients to zero
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < dW1_.rows(); ++i) {
        for (size_t j = 0; j < dW1_.cols(); ++j) {
            dW1_(i, j) = 0.0f;
        }
    }

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < dW2_.rows(); ++i) {
        for (size_t j = 0; j < dW2_.cols(); ++j) {
            dW2_(i, j) = 0.0f;
        }
    }

    #pragma omp parallel for
    for (size_t i = 0; i < db1_.size(); ++i)
        db1_[i] = 0.0f;
    #pragma omp parallel for
    for (size_t i = 0; i < db2_.size(); ++i)
        db2_[i] = 0.0f;
}

Matrix FeedForward::forward(const Matrix& input) {
    try {
        std::cout << "\n=== FeedForward Dimensions Debug ===" << std::endl;
        std::cout << "Input: " << input.rows() << "x" << input.cols() << std::endl;
        std::cout << "W1: " << w1.rows() << "x" << w1.cols() << std::endl;
        std::cout << "W2: " << w2.rows() << "x" << w2.cols() << std::endl;
        std::cout << "B1: " << b1.size() << std::endl;
        std::cout << "B2: " << b2.size() << std::endl;

#ifdef USE_CUDA
        try {
            // Use CUDA memory manager for efficient memory allocation
            auto& memory_mgr = cuda::MemoryManager::instance();
            // Allocate intermediate results
            Matrix intermediate(input.rows(), w1.cols());
            std::cout << "Intermediate dimensions: " << intermediate.rows() << "x" << intermediate.cols() << std::endl;
            
            cuda::matmul(input, w1, intermediate);
            CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
            
            std::cout << "After first matmul - Intermediate: " << intermediate.rows() << "x" << intermediate.cols() << std::endl;
            std::cout << "Bias dimensions: " << b1.size() << std::endl;
            
            // Apply bias and activation
            intermediate.add_bias(b1);
            CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
            
            std::cout << "After bias addition - Intermediate: " << intermediate.rows() << "x" << intermediate.cols() << std::endl;
            cuda::gelu_forward(intermediate);
            CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
            
            std::cout << "After GELU - Intermediate: " << intermediate.rows() << "x" << intermediate.cols() << std::endl;
            // Store for backward pass
            intermediate_cache = intermediate;
            std::cout << "Intermediate cache dimensions: " << intermediate_cache.rows() << "x" << intermediate_cache.cols() << std::endl;
            
            // Explicitly preserve input batch size
            Matrix output(input.rows(), w2.cols());  // Force output to be 1019 x hidden_size
            std::cout << "Output dimensions: " << output.rows() << "x" << output.cols() << std::endl;
            
            cuda::matmul(intermediate, w2, output);
            CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
            
            std::cout << "After second matmul - Output: " << output.rows() << "x" << output.cols() << std::endl;
            output.add_bias(b2);
            CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
            
            std::cout << "After bias addition - Output: " << output.rows() << "x" << output.cols() << std::endl;
            // Check dimensions before residual connection
            if (output.rows() != input.rows() || output.cols() != input.cols()) {
                throw std::runtime_error("FeedForward output dimensions " + 
                    std::to_string(output.rows()) + "x" + std::to_string(output.cols()) +
                    " don't match input dimensions " + 
                    std::to_string(input.rows()) + "x" + std::to_string(input.cols()));
            }
            
            // Add residual connection
            output += input;
            CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
            
            return output;
        } catch (const std::runtime_error& e) {
            std::cerr << "CUDA feed forward failed, falling back to CPU: " << e.what() << std::endl;
#endif
            // CPU fallback implementation
            Matrix intermediate = matmul(input, w1);
            intermediate.add_bias(b1);
            intermediate.apply_gelu();
            intermediate_cache = intermediate;
            
            Matrix output = matmul(intermediate, w2);
            output.add_bias(b2);
            
            // Check dimensions before residual connection
            if (output.rows() != input.rows() || output.cols() != input.cols()) {
                throw std::runtime_error("FeedForward output dimensions " + 
                    std::to_string(output.rows()) + "x" + std::to_string(output.cols()) +
                    " don't match input dimensions " + 
                    std::to_string(input.rows()) + "x" + std::to_string(input.cols()));
            }
            
            // Add residual connection
            output += input;  // This is where the dimension mismatch occurs
            
            return output;
#ifdef USE_CUDA
        }
#endif
    } catch (const std::exception& e) {
        throw std::runtime_error("FeedForward forward failed: " + std::string(e.what()));
    }
}

void FeedForward::save(std::ostream& os) const {
    size_t hidden_size = w2.cols();
    size_t intermediate_size = w1.cols();

    os.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
    os.write(reinterpret_cast<const char*>(&intermediate_size), sizeof(intermediate_size));
    os.write(reinterpret_cast<const char*>(&dropout_prob), sizeof(dropout_prob));

    os.write(reinterpret_cast<const char*>(w1.data()), w1.rows() * w1.cols() * sizeof(float));
    os.write(reinterpret_cast<const char*>(w2.data()), w2.rows() * w2.cols() * sizeof(float));
    os.write(reinterpret_cast<const char*>(b1.data()), b1.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(b2.data()), b2.size() * sizeof(float));
}

std::unique_ptr<FeedForward> FeedForward::load(std::istream& is) {
    size_t hidden_size, intermediate_size;
    float dropout_prob;

    is.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
    is.read(reinterpret_cast<char*>(&intermediate_size), sizeof(intermediate_size));
    is.read(reinterpret_cast<char*>(&dropout_prob), sizeof(dropout_prob));

    auto ffn = std::make_unique<FeedForward>(hidden_size, intermediate_size, dropout_prob);

    is.read(reinterpret_cast<char*>(ffn->w1.data()),
            ffn->w1.rows() * ffn->w1.cols() * sizeof(float));
    is.read(reinterpret_cast<char*>(ffn->w2.data()),
            ffn->w2.rows() * ffn->w2.cols() * sizeof(float));
    is.read(reinterpret_cast<char*>(ffn->b1.data()), ffn->b1.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(ffn->b2.data()), ffn->b2.size() * sizeof(float));

    return ffn;
}

Matrix FeedForward::backward(const Matrix& grad_output, const Matrix& input) {
    std::cout << "FeedForward::backward dimensions:" << std::endl;
    std::cout << "grad_output: " << grad_output.rows() << "x" << grad_output.cols() << std::endl;
    std::cout << "input: " << input.rows() << "x" << input.cols() << std::endl;
    std::cout << "intermediate_cache: " << intermediate_cache.rows() << "x" << intermediate_cache.cols() << std::endl;

    try {
#ifdef USE_CUDA
        // Compute gradients for second layer
        Matrix d_intermediate(grad_output.rows(), w2.rows());  // [batch_size x intermediate_size]
        std::cout << "d_intermediate dims: " << d_intermediate.rows() << "x" << d_intermediate.cols() << std::endl;
        
        cuda::matmul(grad_output, w2.transpose(), d_intermediate);
        
        // Compute gradients for GELU activation
        Matrix gelu_grad = intermediate_cache;  // Create copy for in-place modification
        cuda::gelu_backward(gelu_grad, d_intermediate);  // Compute GELU gradient in-place
        
        if (d_intermediate.rows() != gelu_grad.rows() || d_intermediate.cols() != gelu_grad.cols()) {
            throw std::runtime_error("Dimension mismatch in GELU backward: " + 
                std::to_string(d_intermediate.rows()) + "x" + std::to_string(d_intermediate.cols()) +
                " vs " + std::to_string(gelu_grad.rows()) + "x" + std::to_string(gelu_grad.cols()));
        }
        d_intermediate = d_intermediate.hadamard(gelu_grad);
        
        // Compute input gradients
        Matrix d_input(input.rows(), input.cols());  // [batch_size x hidden_size]
        std::cout << "d_input dims before matmul: " << d_input.rows() << "x" << d_input.cols() << std::endl;
        cuda::matmul(d_intermediate, w1.transpose(), d_input);
        std::cout << "d_input dims after matmul: " << d_input.rows() << "x" << d_input.cols() << std::endl;
        
        // Verify output dimensions match input dimensions
        if (d_input.rows() != input.rows() || d_input.cols() != input.cols()) {
            throw std::runtime_error("Output matrix has wrong dimensions: expected " +
                std::to_string(input.rows()) + "x" + std::to_string(input.cols()) +
                " got " + std::to_string(d_input.rows()) + "x" + std::to_string(d_input.cols()));
        }
        
        return d_input;
#else
        throw std::runtime_error("CUDA support not enabled");
#endif
    } catch (const std::exception& e) {
        throw std::runtime_error("FeedForward backward failed: " + std::string(e.what()));
    }
}

void FeedForward::update_parameters(const Matrix& grad) {
    float learning_rate = 0.01f;  // Could be made configurable
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < w1.rows(); ++i) {
        for (size_t j = 0; j < w1.cols(); ++j) {
            w1(i, j) -= dW1_(i, j) * learning_rate;
        }
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < b1.size(); ++i) {
        b1[i] -= db1_[i] * learning_rate;
    }
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < w2.rows(); ++i) {
        for (size_t j = 0; j < w2.cols(); ++j) {
            w2(i, j) -= dW2_(i, j) * learning_rate;
        }
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < b2.size(); ++i) {
        b2[i] -= db2_[i] * learning_rate;
    }
}

void FeedForward::initialize_weights() {
    // Get sizes from weight matrices
    size_t hidden_size = w1.rows();  // Input/output size
    size_t intermediate_size = w1.cols();  // Hidden layer size
    
    float scale = sqrt(2.0f / (hidden_size + intermediate_size));
    
    w1.initialize_random(scale);
    w2.initialize_random(scale);
    
    // Initialize biases to small non-zero values
    b1.initialize_constant(0.01f);
    b2.initialize_constant(0.01f);
}