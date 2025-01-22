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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

FeedForward::FeedForward(size_t hidden_size, size_t intermediate_size, float dropout)
    : w1(hidden_size, intermediate_size), w2(intermediate_size, hidden_size), b1(intermediate_size),
      b2(hidden_size), dropout_prob(dropout), intermediate_cache(1, intermediate_size),
      // Initialize gradients with same dimensions as their parameters
      dW1_(hidden_size, intermediate_size), dW2_(intermediate_size, hidden_size),
      db1_(intermediate_size), db2_(hidden_size) {

    std::cout << "\n=== FeedForward Constructor Dimensions ===" << std::endl;
    std::cout << "Hidden size: " << hidden_size << std::endl;
    std::cout << "Intermediate size: " << intermediate_size << std::endl;
    std::cout << "w1: " << w1.rows() << "x" << w1.cols() << std::endl;
    std::cout << "w2: " << w2.rows() << "x" << w2.cols() << std::endl;
    std::cout << "b1 size: " << b1.size() << std::endl;
    std::cout << "b2 size: " << b2.size() << std::endl;

    // Initialize weights with Xavier/Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());

    float w1_limit = std::sqrt(6.0f / (hidden_size + intermediate_size));
    float w2_limit = std::sqrt(6.0f / (intermediate_size + hidden_size));

    std::uniform_real_distribution<float> w1_dis(-w1_limit, w1_limit);
    std::uniform_real_distribution<float> w2_dis(-w2_limit, w2_limit);

    // Initialize weights
    for (size_t i = 0; i < w1.rows(); ++i) {
        for (size_t j = 0; j < w1.cols(); ++j) {
            w1(i, j) = w1_dis(gen);
        }
    }

    for (size_t i = 0; i < w2.rows(); ++i) {
        for (size_t j = 0; j < w2.cols(); ++j) {
            w2(i, j) = w2_dis(gen);
        }
    }

    // Initialize biases to zero
    for (size_t i = 0; i < b1.size(); ++i)
        b1[i] = 0.0f;
    for (size_t i = 0; i < b2.size(); ++i)
        b2[i] = 0.0f;

    // Initialize gradients to zero
    for (size_t i = 0; i < dW1_.rows(); ++i) {
        for (size_t j = 0; j < dW1_.cols(); ++j) {
            dW1_(i, j) = 0.0f;
        }
    }

    for (size_t i = 0; i < dW2_.rows(); ++i) {
        for (size_t j = 0; j < dW2_.cols(); ++j) {
            dW2_(i, j) = 0.0f;
        }
    }

    for (size_t i = 0; i < db1_.size(); ++i)
        db1_[i] = 0.0f;
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
            cuda::matmul(input, w1, intermediate);
            std::cout << "After first matmul - Intermediate: " << intermediate.rows() << "x" << intermediate.cols() << std::endl;
            
            // Apply bias and activation
            intermediate.add_bias(b1);
            cuda::gelu_forward(intermediate);
            
            // Store for backward pass
            intermediate_cache = intermediate;
            
            // Explicitly preserve input batch size
            Matrix output(input.rows(), w2.cols());  // Force output to be 1019 x hidden_size
            cuda::matmul(intermediate, w2, output);
            std::cout << "After second matmul - Output: " << output.rows() << "x" << output.cols() << std::endl;
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

Matrix FeedForward::backward_cuda(const Matrix& grad_output, const Matrix& input) const {
#ifdef USE_CUDA
    return backward_cuda(grad_output, input);
#else
    throw std::runtime_error("CUDA support not enabled");
#endif
}

void FeedForward::update_parameters(const Matrix& grad) {
    float learning_rate = 0.01f;  // Could be made configurable
    
    w1 -= dW1_ * learning_rate;
    // Scale vector elements individually
    for (size_t i = 0; i < b1.size(); ++i) {
        b1[i] -= db1_[i] * learning_rate;
    }
    
    w2 -= dW2_ * learning_rate;
    // Scale vector elements individually
    for (size_t i = 0; i < b2.size(); ++i) {
        b2[i] -= db2_[i] * learning_rate;
    }
}