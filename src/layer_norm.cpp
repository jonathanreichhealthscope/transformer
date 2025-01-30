#include "../include/layer_norm.hpp"
#include "../include/cuda/matrix_ops.cuh"
#include <cmath>
#include <omp.h>
#include "../include/cuda/backward_ops.cuh"

LayerNorm::LayerNorm(size_t hidden_size_, float eps_)
    : hidden_size_(hidden_size_),
      eps_(eps_) {
    // Initialize parameters
    params_.gamma = Matrix(1, hidden_size_, 1.0f);  // Initialize to ones
    params_.beta = Matrix(1, hidden_size_, 0.0f);   // Initialize to zeros

    // Initialize gradients
    grads_.gamma_grad = Matrix(1, hidden_size_);
    grads_.beta_grad = Matrix(1, hidden_size_);
}

Matrix LayerNorm::forward(const Matrix& input) {
    try {
        std::cout << "\n=== LayerNorm::forward START ===" << std::endl;
        std::cout << "Input dims: " << input.rows() << "x" << input.cols() << std::endl;
        std::cout << "Expected hidden_size: " << hidden_size_ << std::endl;
        
        if (input.cols() != hidden_size_) {
            std::cout << "Input shape: " << input.rows() << "x" << input.cols() << std::endl;
            std::cout << "Gamma shape: " << params_.gamma.rows() << "x" << params_.gamma.cols() << std::endl;
            throw std::runtime_error("Input dimension mismatch in LayerNorm");
        }

        input_cache_ = input;
        
        Matrix mean(input.rows(), 1, 0.0f);
        Matrix var(input.rows(), 1, 0.0f);
        std::cout << "Mean dims: " << mean.rows() << "x" << mean.cols() << std::endl;
        std::cout << "Var dims: " << var.rows() << "x" << var.cols() << std::endl;

        // Compute mean for each row
        #pragma omp parallel for
        for (size_t i = 0; i < input.rows(); i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < input.cols(); j++) {
                sum += input(i, j);
            }
            mean(i, 0) = sum / input.cols();
        }

        // Compute variance for each row
        #pragma omp parallel for
        for (size_t i = 0; i < input.rows(); i++) {
            float sum_sq = 0.0f;
            for (size_t j = 0; j < input.cols(); j++) {
                float diff = input(i, j) - mean(i, 0);
                sum_sq += diff * diff;
            }
            var(i, 0) = sum_sq / input.cols();
        }

        Matrix output(input.rows(), input.cols());
        std::cout << "Output dims: " << output.rows() << "x" << output.cols() << std::endl;
        std::cout << "Gamma dims: " << params_.gamma.rows() << "x" << params_.gamma.cols() << std::endl;
        std::cout << "Beta dims: " << params_.beta.rows() << "x" << params_.beta.cols() << std::endl;

        // Normalize and apply scale/shift
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < input.rows(); i++) {
            for (size_t j = 0; j < input.cols(); j++) {
                float normalized = (input(i, j) - mean(i, 0)) / std::sqrt(var(i, 0) + eps_);
                output(i, j) = params_.gamma(0, j) * normalized + params_.beta(0, j);
            }
        }

        std::cout << "=== LayerNorm::forward END ===\n" << std::endl;
        return output;

    } catch (const std::exception& e) {
        throw std::runtime_error("Error in LayerNorm forward: " + std::string(e.what()));
    }
}

Matrix LayerNorm::compute_gradients(const Matrix& grad_output) {
    try {
        std::cout << "\n=== LayerNorm::compute_gradients START ===" << std::endl;
        std::cout << "grad_output dims: " << grad_output.rows() << "x" << grad_output.cols() << std::endl;
        std::cout << "input_cache_ dims: " << input_cache_.rows() << "x" << input_cache_.cols() << std::endl;

        Matrix grad_input(input_cache_.rows(), input_cache_.cols());
        std::cout << "grad_input dims: " << grad_input.rows() << "x" << grad_input.cols() << std::endl;

        Matrix mean(input_cache_.rows(), 1);
        Matrix var(input_cache_.rows(), 1);
        std::cout << "mean dims: " << mean.rows() << "x" << mean.cols() << std::endl;
        std::cout << "var dims: " << var.rows() << "x" << var.cols() << std::endl;

        // Compute mean and variance
        #pragma omp parallel for
        for (size_t i = 0; i < input_cache_.rows(); i++) {
            float sum = 0.0f;
            float sum_sq = 0.0f;
            for (size_t j = 0; j < input_cache_.cols(); j++) {
                sum += input_cache_(i, j);
            }
            mean(i, 0) = sum / input_cache_.cols();

            for (size_t j = 0; j < input_cache_.cols(); j++) {
                float diff = input_cache_(i, j) - mean(i, 0);
                sum_sq += diff * diff;
            }
            var(i, 0) = sum_sq / input_cache_.cols();
        }

        // Compute gradients
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < input_cache_.rows(); i++) {
            for (size_t j = 0; j < input_cache_.cols(); j++) {
                float inv_std = 1.0f / std::sqrt(var(i, 0) + eps_);
                float normalized = (input_cache_(i, j) - mean(i, 0)) * inv_std;
                
                // Gradient with respect to input
                grad_input(i, j) = params_.gamma(0, j) * grad_output(i, j) * inv_std;
                
                // Accumulate gradients for gamma and beta
                #pragma omp atomic
                grads_.gamma_grad(0, j) += grad_output(i, j) * normalized;
                #pragma omp atomic
                grads_.beta_grad(0, j) += grad_output(i, j);
            }
        }

        std::cout << "gamma_grad dims: " << grads_.gamma_grad.rows() << "x" << grads_.gamma_grad.cols() << std::endl;
        std::cout << "beta_grad dims: " << grads_.beta_grad.rows() << "x" << grads_.beta_grad.cols() << std::endl;
        std::cout << "=== LayerNorm::compute_gradients END ===\n" << std::endl;

        return grad_input;

    } catch (const std::exception& e) {
        throw std::runtime_error("Error in LayerNorm backward: " + std::string(e.what()));
    }
}

void LayerNorm::save(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(params_.gamma.data()), hidden_size_ * sizeof(float));
    os.write(reinterpret_cast<const char*>(params_.beta.data()), hidden_size_ * sizeof(float));
}

std::unique_ptr<LayerNorm> LayerNorm::load(std::istream& is) {
    size_t hidden_size;
    is.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
    
    auto ln = std::make_unique<LayerNorm>(hidden_size);
    
    // Read parameters
    is.read(reinterpret_cast<char*>(ln->params_.gamma.data()), hidden_size * sizeof(float));
    is.read(reinterpret_cast<char*>(ln->params_.beta.data()), hidden_size * sizeof(float));
    
    return ln;
}