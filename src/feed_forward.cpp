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
        std::cout << "\n=== FeedForward::forward START ===" << std::endl;
        std::cout << "Input dims: " << input.rows() << "x" << input.cols() << std::endl;
        std::cout << "FF1 weights dims: " << params_.ff1_weights.rows() << "x" << params_.ff1_weights.cols() << std::endl;
        std::cout << "FF2 weights dims: " << params_.ff2_weights.rows() << "x" << params_.ff2_weights.cols() << std::endl;
        std::cout << "FF1 bias size: " << params_.ff1_bias.size() << std::endl;
        std::cout << "FF2 bias size: " << params_.ff2_bias.size() << std::endl;

        // First layer
        Matrix intermediate = matmul(input, params_.ff1_weights);
        std::cout << "After FF1 dims: " << intermediate.rows() << "x" << intermediate.cols() << std::endl;

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
        std::cout << "After ReLU dims: " << intermediate.rows() << "x" << intermediate.cols() << std::endl;

        // Cache intermediate values
        intermediate_cache = intermediate;
        input_cache_ = input;
        std::cout << "Cached intermediate dims: " << intermediate_cache.rows() << "x" << intermediate_cache.cols() << std::endl;
        std::cout << "Cached input dims: " << input_cache_.rows() << "x" << input_cache_.cols() << std::endl;

        // Second layer
        Matrix output = matmul(intermediate, params_.ff2_weights);
        std::cout << "Final output dims: " << output.rows() << "x" << output.cols() << std::endl;
        
        for (size_t i = 0; i < output.rows(); ++i) {
            for (size_t j = 0; j < output.cols(); ++j) {
                output(i, j) += params_.ff2_bias[j];
            }
        }

        std::cout << "=== FeedForward::forward END ===\n" << std::endl;
        return output;

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
        std::cout << "\n=== FeedForward::backward START ===" << std::endl;
        std::cout << "grad_output dims: " << grad_output.rows() << "x" << grad_output.cols() << std::endl;
        std::cout << "input dims: " << input.rows() << "x" << input.cols() << std::endl;
        std::cout << "ff2_weights dims: " << params_.ff2_weights.rows() << "x" << params_.ff2_weights.cols() << std::endl;
        std::cout << "ff1_weights dims: " << params_.ff1_weights.rows() << "x" << params_.ff1_weights.cols() << std::endl;
        std::cout << "intermediate_cache dims: " << intermediate_cache.rows() << "x" << intermediate_cache.cols() << std::endl;
        std::cout << "input_cache_ dims: " << input_cache_.rows() << "x" << input_cache_.cols() << std::endl;

        Matrix w2_transpose = params_.ff2_weights.transpose();
        std::cout << "w2_transpose dims: " << w2_transpose.rows() << "x" << w2_transpose.cols() << std::endl;

        Matrix d_intermediate = matmul(grad_output, w2_transpose);
        std::cout << "d_intermediate dims: " << d_intermediate.rows() << "x" << d_intermediate.cols() << std::endl;

        // Apply ReLU gradient
        for (size_t i = 0; i < intermediate_cache.rows(); ++i) {
            for (size_t j = 0; j < intermediate_cache.cols(); ++j) {
                if (intermediate_cache(i, j) <= 0) {
                    d_intermediate(i, j) = 0;
                }
            }
        }

        Matrix w1_transpose = params_.ff1_weights.transpose();
        std::cout << "w1_transpose dims: " << w1_transpose.rows() << "x" << w1_transpose.cols() << std::endl;

        Matrix d_input = matmul(d_intermediate, w1_transpose);
        std::cout << "d_input dims: " << d_input.rows() << "x" << d_input.cols() << std::endl;

        // Update gradients
        grads_.ff1_grad = matmul(input.transpose(), d_intermediate);
        grads_.ff2_grad = matmul(intermediate_cache.transpose(), grad_output);
        std::cout << "ff1_grad dims: " << grads_.ff1_grad.rows() << "x" << grads_.ff1_grad.cols() << std::endl;
        std::cout << "ff2_grad dims: " << grads_.ff2_grad.rows() << "x" << grads_.ff2_grad.cols() << std::endl;

        std::cout << "=== FeedForward::backward END ===\n" << std::endl;
        return d_input;

    } catch (const std::exception& e) {
        std::cerr << "\nError in FeedForward::backward: " << e.what() << std::endl;
        throw;
    }
}

void FeedForward::update_parameters(const Matrix& grad, float learning_rate) {
    const float max_grad_norm = 5.0f;  // Match global clipping threshold
    
    // Compute gradient norms
    float grad_norm_ff1 = 0.0f;
    #pragma omp parallel for reduction(+:grad_norm_ff1)
    for (size_t i = 0; i < grads_.ff1_grad.size(); ++i) {
        grad_norm_ff1 += grads_.ff1_grad.data()[i] * grads_.ff1_grad.data()[i];
    }
    grad_norm_ff1 = std::sqrt(grad_norm_ff1);
    
    float grad_norm_ff2 = 0.0f;
    #pragma omp parallel for reduction(+:grad_norm_ff2)
    for (size_t i = 0; i < grads_.ff2_grad.size(); ++i) {
        grad_norm_ff2 += grads_.ff2_grad.data()[i] * grads_.ff2_grad.data()[i];
    }
    grad_norm_ff2 = std::sqrt(grad_norm_ff2);
    
    // Compute scaling factors
    float scale_ff1 = std::min(max_grad_norm / (grad_norm_ff1 + 1e-6f), 1.0f);
    float scale_ff2 = std::min(max_grad_norm / (grad_norm_ff2 + 1e-6f), 1.0f);
    
    // Debug output
    std::cout << "FF Gradient norms - FF1: " << grad_norm_ff1 << ", FF2: " << grad_norm_ff2 << std::endl;
    std::cout << "FF Scaling factors - FF1: " << scale_ff1 << ", FF2: " << scale_ff2 << std::endl;
    std::cout << "Learning rate: " << learning_rate << std::endl;
    
    // Update first layer weights with clipping
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < params_.ff1_weights.rows(); ++i) {
        for (size_t j = 0; j < params_.ff1_weights.cols(); ++j) {
            float clipped_grad = grads_.ff1_grad(i, j) * scale_ff1;
            params_.ff1_weights(i, j) -= clipped_grad * learning_rate;
        }
    }
    
    // Update first layer bias with clipping
    #pragma omp parallel for
    for (size_t i = 0; i < params_.ff1_bias.size(); ++i) {
        float clipped_grad = grads_.ff1_bias_grad[i] * scale_ff1;
        params_.ff1_bias[i] -= clipped_grad * learning_rate;
    }
    
    // Update second layer weights with clipping
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < params_.ff2_weights.rows(); ++i) {
        for (size_t j = 0; j < params_.ff2_weights.cols(); ++j) {
            float clipped_grad = grads_.ff2_grad(i, j) * scale_ff2;
            params_.ff2_weights(i, j) -= clipped_grad * learning_rate;
        }
    }
    
    // Update second layer bias with clipping
    #pragma omp parallel for
    for (size_t i = 0; i < params_.ff2_bias.size(); ++i) {
        float clipped_grad = grads_.ff2_bias_grad[i] * scale_ff2;
        params_.ff2_bias[i] -= clipped_grad * learning_rate;
    }
    
    // Zero out gradients after update
    grads_.ff1_grad.fill(0.0f);
    grads_.ff2_grad.fill(0.0f);
    std::fill(grads_.ff1_bias_grad.begin(), grads_.ff1_bias_grad.end(), 0.0f);
    std::fill(grads_.ff2_bias_grad.begin(), grads_.ff2_bias_grad.end(), 0.0f);
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