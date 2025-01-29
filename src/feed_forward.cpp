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
    : dropout_prob(dropout), intermediate_cache(1, intermediate_size) {
    
    // Initialize matrices with correct dimensions
    params_.ff1_weights = Matrix(hidden_size, intermediate_size);
    params_.ff2_weights = Matrix(intermediate_size, hidden_size);
    params_.ff1_bias = FloatVector(intermediate_size);
    params_.ff2_bias = FloatVector(hidden_size);
    
    // Initialize gradients with same dimensions
    grads_.ff1_grad = Matrix(hidden_size, intermediate_size);
    grads_.ff2_grad = Matrix(intermediate_size, hidden_size);
    grads_.ff1_bias_grad = FloatVector(intermediate_size);
    grads_.ff2_bias_grad = FloatVector(hidden_size);

    // Initialize weights with Xavier/Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());

    float w1_limit = std::sqrt(6.0f / (hidden_size + intermediate_size));
    float w2_limit = std::sqrt(6.0f / (intermediate_size + hidden_size));

    std::uniform_real_distribution<float> w1_dis(-w1_limit, w1_limit);
    std::uniform_real_distribution<float> w2_dis(-w2_limit, w2_limit);

    // Initialize weights
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < params_.ff1_weights.rows(); ++i) {
        for (size_t j = 0; j < params_.ff1_weights.cols(); ++j) {
            params_.ff1_weights(i, j) = w1_dis(gen);
        }
    }

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < params_.ff2_weights.rows(); ++i) {
        for (size_t j = 0; j < params_.ff2_weights.cols(); ++j) {
            params_.ff2_weights(i, j) = w2_dis(gen);
        }
    }

    // Initialize biases to zero
    #pragma omp parallel for
    for (size_t i = 0; i < params_.ff1_bias.size(); ++i)
        params_.ff1_bias[i] = 0.0f;
    #pragma omp parallel for
    for (size_t i = 0; i < params_.ff2_bias.size(); ++i)
        params_.ff2_bias[i] = 0.0f;

    // Initialize gradients to zero
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < grads_.ff1_grad.rows(); ++i) {
        for (size_t j = 0; j < grads_.ff1_grad.cols(); ++j) {
            grads_.ff1_grad(i, j) = 0.0f;
        }
    }

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < grads_.ff2_grad.rows(); ++i) {
        for (size_t j = 0; j < grads_.ff2_grad.cols(); ++j) {
            grads_.ff2_grad(i, j) = 0.0f;
        }
    }

    #pragma omp parallel for
    for (size_t i = 0; i < grads_.ff1_bias_grad.size(); ++i)
        grads_.ff1_bias_grad[i] = 0.0f;
    #pragma omp parallel for
    for (size_t i = 0; i < grads_.ff2_bias_grad.size(); ++i)
        grads_.ff2_bias_grad[i] = 0.0f;
}

Matrix FeedForward::forward(const Matrix& input) {
    try {
        // Debug dimensions
        std::cout << "Input: " << input.rows() << "x" << input.cols() << std::endl;
        std::cout << "W1: " << params_.ff1_weights.rows() << "x" << params_.ff1_weights.cols() << std::endl;
        std::cout << "W2: " << params_.ff2_weights.rows() << "x" << params_.ff2_weights.cols() << std::endl;
        std::cout << "B1: " << params_.ff1_bias.size() << std::endl;
        std::cout << "B2: " << params_.ff2_bias.size() << std::endl;

#ifdef USE_CUDA
        try {
            auto& memory_mgr = cuda::MemoryManager::instance();
            // TODO: Implement CUDA path
            throw std::runtime_error("CUDA implementation not yet available");
        } catch (const std::runtime_error& e) {
            std::cerr << "CUDA forward failed, falling back to CPU: " << e.what() << std::endl;
#endif
            // CPU implementation
            // First layer
            Matrix intermediate = matmul(input, params_.ff1_weights);
            for (size_t i = 0; i < intermediate.rows(); ++i) {
                for (size_t j = 0; j < intermediate.cols(); ++j) {
                    intermediate(i, j) += params_.ff1_bias[j];
                }
            }

            // Apply ReLU
            for (size_t i = 0; i < intermediate.rows(); ++i) {
                for (size_t j = 0; j < intermediate.cols(); ++j) {
                    intermediate(i, j) = std::max(0.0f, intermediate(i, j));
                }
            }

            // Cache intermediate values for backward pass
            intermediate_cache = intermediate;
            input_cache_ = input;

            // Second layer
            Matrix output = matmul(intermediate, params_.ff2_weights);
            for (size_t i = 0; i < output.rows(); ++i) {
                for (size_t j = 0; j < output.cols(); ++j) {
                    output(i, j) += params_.ff2_bias[j];
                }
            }

            return output;
#ifdef USE_CUDA
        }
#endif
    } catch (const std::exception& e) {
        throw std::runtime_error("FeedForward forward failed: " + std::string(e.what()));
    }
}

void FeedForward::save(std::ostream& os) const {
    size_t hidden_size = params_.ff2_weights.cols();
    size_t intermediate_size = params_.ff1_weights.cols();
    
    os.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
    os.write(reinterpret_cast<const char*>(&intermediate_size), sizeof(intermediate_size));
    
    // Save weights
    os.write(reinterpret_cast<const char*>(params_.ff1_weights.data()), 
             params_.ff1_weights.rows() * params_.ff1_weights.cols() * sizeof(float));
    os.write(reinterpret_cast<const char*>(params_.ff2_weights.data()), 
             params_.ff2_weights.rows() * params_.ff2_weights.cols() * sizeof(float));
    
    // Save biases
    os.write(reinterpret_cast<const char*>(params_.ff1_bias.data()), params_.ff1_bias.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(params_.ff2_bias.data()), params_.ff2_bias.size() * sizeof(float));
}

std::unique_ptr<FeedForward> FeedForward::load(std::istream& is) {
    size_t hidden_size, intermediate_size;
    is.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
    is.read(reinterpret_cast<char*>(&intermediate_size), sizeof(intermediate_size));
    
    auto ffn = std::make_unique<FeedForward>(hidden_size, intermediate_size);
    
    // Load weights
    is.read(reinterpret_cast<char*>(ffn->params_.ff1_weights.data()),
            ffn->params_.ff1_weights.rows() * ffn->params_.ff1_weights.cols() * sizeof(float));
    is.read(reinterpret_cast<char*>(ffn->params_.ff2_weights.data()),
            ffn->params_.ff2_weights.rows() * ffn->params_.ff2_weights.cols() * sizeof(float));
    
    // Load biases
    is.read(reinterpret_cast<char*>(ffn->params_.ff1_bias.data()), ffn->params_.ff1_bias.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(ffn->params_.ff2_bias.data()), ffn->params_.ff2_bias.size() * sizeof(float));
    
    return ffn;
}

Matrix FeedForward::backward(const Matrix& grad_output, const Matrix& input) {
    try {
        // Initial debug dimensions
        std::cout << "\n=== FeedForward::backward START ===" << std::endl;
        std::cout << "Input dimensions:" << std::endl;
        std::cout << "grad_output: " << grad_output.rows() << "x" << grad_output.cols() << std::endl;
        std::cout << "input: " << input.rows() << "x" << input.cols() << std::endl;
        std::cout << "ff2_weights: " << params_.ff2_weights.rows() << "x" << params_.ff2_weights.cols() << std::endl;
        std::cout << "ff1_weights: " << params_.ff1_weights.rows() << "x" << params_.ff1_weights.cols() << std::endl;
        std::cout << "intermediate_cache: " << intermediate_cache.rows() << "x" << intermediate_cache.cols() << std::endl;
        std::cout << "input_cache_: " << input_cache_.rows() << "x" << input_cache_.cols() << std::endl;

        // Compute gradients for second layer
        std::cout << "\nComputing second layer gradients..." << std::endl;
        
        // Create transposed weight matrix
        Matrix w2_transpose = params_.ff2_weights.transpose();
        std::cout << "w2_transpose dimensions: " << w2_transpose.rows() << "x" << w2_transpose.cols() << std::endl;
        
        // Initialize d_intermediate with correct dimensions
        Matrix d_intermediate(grad_output.rows(), w2_transpose.cols());
        std::cout << "d_intermediate initialized with dimensions: " << d_intermediate.rows() << "x" << d_intermediate.cols() << std::endl;
        
        // Compute gradient through second layer
        std::cout << "Computing matmul(grad_output, w2_transpose)..." << std::endl;
        d_intermediate = matmul(grad_output, w2_transpose);
        std::cout << "d_intermediate after matmul dimensions: " << d_intermediate.rows() << "x" << d_intermediate.cols() << std::endl;
        
        // Update second layer gradients
        std::cout << "\nComputing second layer weight gradients..." << std::endl;
        std::cout << "intermediate_cache transpose dimensions: " << intermediate_cache.rows() << "x" << intermediate_cache.cols() << std::endl;
        grads_.ff2_grad = matmul(intermediate_cache.transpose(), grad_output);
        std::cout << "ff2_grad dimensions: " << grads_.ff2_grad.rows() << "x" << grads_.ff2_grad.cols() << std::endl;
        
        // Accumulate bias gradients for second layer
        std::cout << "\nAccumulating second layer bias gradients..." << std::endl;
        if (grads_.ff2_bias_grad.size() != grad_output.cols()) {
            std::cout << "Resizing ff2_bias_grad from " << grads_.ff2_bias_grad.size() 
                      << " to " << grad_output.cols() << std::endl;
            grads_.ff2_bias_grad = FloatVector(grad_output.cols());
        }
        
        for (size_t j = 0; j < grad_output.cols(); ++j) {
            float bias_grad = 0.0f;
            for (size_t i = 0; i < grad_output.rows(); ++i) {
                bias_grad += grad_output(i, j);
            }
            grads_.ff2_bias_grad[j] = bias_grad;
        }
        std::cout << "ff2_bias_grad size: " << grads_.ff2_bias_grad.size() << std::endl;

        // Apply ReLU gradient
        std::cout << "\nApplying ReLU gradient..." << std::endl;
        std::cout << "Before ReLU gradient - d_intermediate dimensions: " << d_intermediate.rows() << "x" << d_intermediate.cols() << std::endl;
        for (size_t i = 0; i < intermediate_cache.rows(); ++i) {
            for (size_t j = 0; j < intermediate_cache.cols(); ++j) {
                if (intermediate_cache(i, j) <= 0) {
                    d_intermediate(i, j) = 0;
                }
            }
        }
        std::cout << "After ReLU gradient - d_intermediate dimensions: " << d_intermediate.rows() << "x" << d_intermediate.cols() << std::endl;

        // Compute gradients for first layer
        std::cout << "\nComputing first layer gradients..." << std::endl;
        Matrix w1_transpose = params_.ff1_weights.transpose();
        std::cout << "w1_transpose dimensions: " << w1_transpose.rows() << "x" << w1_transpose.cols() << std::endl;
        
        // Initialize d_input with correct dimensions
        Matrix d_input(d_intermediate.rows(), w1_transpose.cols());
        std::cout << "d_input initialized with dimensions: " << d_input.rows() << "x" << d_input.cols() << std::endl;
        
        // Compute input gradients
        std::cout << "Computing matmul(d_intermediate, w1_transpose)..." << std::endl;
        d_input = matmul(d_intermediate, w1_transpose);
        std::cout << "d_input after matmul dimensions: " << d_input.rows() << "x" << d_input.cols() << std::endl;
        
        // Update first layer gradients
        std::cout << "\nComputing first layer weight gradients..." << std::endl;
        grads_.ff1_grad = matmul(input.transpose(), d_intermediate);
        std::cout << "ff1_grad dimensions: " << grads_.ff1_grad.rows() << "x" << grads_.ff1_grad.cols() << std::endl;
        
        // Accumulate bias gradients for first layer
        std::cout << "\nAccumulating first layer bias gradients..." << std::endl;
        if (grads_.ff1_bias_grad.size() != d_intermediate.cols()) {
            std::cout << "Resizing ff1_bias_grad from " << grads_.ff1_bias_grad.size() 
                      << " to " << d_intermediate.cols() << std::endl;
            grads_.ff1_bias_grad = FloatVector(d_intermediate.cols());
        }
        
        for (size_t j = 0; j < d_intermediate.cols(); ++j) {
            float bias_grad = 0.0f;
            for (size_t i = 0; i < d_intermediate.rows(); ++i) {
                bias_grad += d_intermediate(i, j);
            }
            grads_.ff1_bias_grad[j] = bias_grad;
        }
        std::cout << "ff1_bias_grad size: " << grads_.ff1_bias_grad.size() << std::endl;

        std::cout << "\n=== FeedForward::backward END ===" << std::endl;
        return d_input;
    } catch (const std::exception& e) {
        std::cerr << "\nError in FeedForward::backward: " << e.what() << std::endl;
        std::cerr << "Last known dimensions:" << std::endl;
        std::cerr << "grad_output: " << grad_output.rows() << "x" << grad_output.cols() << std::endl;
        std::cerr << "input: " << input.rows() << "x" << input.cols() << std::endl;
        std::cerr << "ff2_weights: " << params_.ff2_weights.rows() << "x" << params_.ff2_weights.cols() << std::endl;
        std::cerr << "ff1_weights: " << params_.ff1_weights.rows() << "x" << params_.ff1_weights.cols() << std::endl;
        throw;
    }
}

void FeedForward::update_parameters(const Matrix& grad) {
    float learning_rate = 0.01f;  // Could be made configurable
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < params_.ff1_weights.rows(); ++i) {
        for (size_t j = 0; j < params_.ff1_weights.cols(); ++j) {
            params_.ff1_weights(i, j) -= grads_.ff1_grad(i, j) * learning_rate;
        }
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < params_.ff1_bias.size(); ++i) {
        params_.ff1_bias[i] -= grads_.ff1_bias_grad[i] * learning_rate;
    }
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < params_.ff2_weights.rows(); ++i) {
        for (size_t j = 0; j < params_.ff2_weights.cols(); ++j) {
            params_.ff2_weights(i, j) -= grads_.ff2_grad(i, j) * learning_rate;
        }
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < params_.ff2_bias.size(); ++i) {
        params_.ff2_bias[i] -= grads_.ff2_bias_grad[i] * learning_rate;
    }
}

void FeedForward::initialize_weights() {
    // Get dimensions
    size_t hidden_size = params_.ff1_weights.rows();
    size_t intermediate_size = params_.ff1_weights.cols();

    // Initialize first layer weights with Xavier/Glorot initialization
    float limit1 = std::sqrt(6.0f / (hidden_size + intermediate_size));
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < params_.ff1_weights.rows(); ++i) {
        for (size_t j = 0; j < params_.ff1_weights.cols(); ++j) {
            params_.ff1_weights(i, j) = (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * limit1;
        }
    }

    // Initialize second layer weights
    float limit2 = std::sqrt(6.0f / (intermediate_size + hidden_size));
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < params_.ff2_weights.rows(); ++i) {
        for (size_t j = 0; j < params_.ff2_weights.cols(); ++j) {
            params_.ff2_weights(i, j) = (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * limit2;
        }
    }

    // Initialize biases to small positive values
    #pragma omp parallel for
    for (size_t i = 0; i < params_.ff1_bias.size(); ++i) {
        params_.ff1_bias[i] = 0.01f;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < params_.ff2_bias.size(); ++i) {
        params_.ff2_bias[i] = 0.01f;
    }
}