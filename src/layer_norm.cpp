#include "../include/layer_norm.hpp"
#include "../include/cuda/matrix_ops.cuh"
#include <cmath>
#include <omp.h>
#include "../include/cuda/backward_ops.cuh"

LayerNorm::LayerNorm(size_t hidden_size, float eps)
    : hidden_size_(hidden_size), 
      eps_(eps),
      gamma_(Matrix(1, hidden_size, 1.0f)),
      beta_(Matrix(1, hidden_size, 0.0f)),
      input_cache_(Matrix(1, hidden_size)),
      output_cache_(Matrix(1, hidden_size)),
      grad_gamma_(Matrix(1, hidden_size)),
      grad_beta_(Matrix(1, hidden_size)) {}

Matrix LayerNorm::forward(const Matrix& input) {
    try {
        input_cache_ = input;  // Store for backward pass
        
#ifdef USE_CUDA
        try {
            Matrix output(input.rows(), input.cols());
            cuda::layer_norm_forward(input, gamma_, beta_, output, eps_);
            return output;
        } catch (const std::runtime_error& e) {
            std::cerr << "CUDA layer norm failed, falling back to CPU: " << e.what() << std::endl;
#endif
            // CPU implementation
            Matrix output(input.rows(), input.cols());
            const float MIN_VAR = 1e-6f;  // Minimum variance threshold
            
            for (size_t i = 0; i < input.rows(); ++i) {
                float mean = 0.0f;
                float var = 0.0f;
                
                // Compute mean
                for (size_t j = 0; j < input.cols(); ++j) {
                    mean += input(i, j);
                }
                mean /= input.cols();
                
                // Compute variance
                for (size_t j = 0; j < input.cols(); ++j) {
                    float diff = input(i, j) - mean;
                    var += diff * diff;
                }
                var = std::max(var / input.cols(), MIN_VAR);  // Apply minimum variance threshold
                
                // Normalize
                float std = std::sqrt(var + eps_);
                for (size_t j = 0; j < input.cols(); ++j) {
                    output(i, j) = gamma_(0, j) * (input(i, j) - mean) / std + beta_(0, j);
                }
            }
            output_cache_ = output;  // Store for backward pass
            return output;
#ifdef USE_CUDA
        }
#endif
    } catch (const std::exception& e) {
        throw std::runtime_error("LayerNorm forward failed: " + std::string(e.what()));
    }
}

Matrix LayerNorm::compute_gradients(const Matrix& grad_output) {
    try {
        Matrix grad_input(input_cache_.rows(), input_cache_.cols());
#ifdef USE_CUDA
        try {
            cuda::layer_norm_backward(grad_output, input_cache_, gamma_, 
                                    grad_gamma_, grad_beta_, eps_);
            return grad_input;
        } catch (const std::runtime_error& e) {
            std::cerr << "CUDA layer norm backward failed, falling back to CPU: " << e.what() << std::endl;
#endif
            // CPU implementation
            grad_gamma_ = Matrix(1, hidden_size_);
            grad_beta_ = Matrix(1, hidden_size_);

            for (size_t i = 0; i < input_cache_.rows(); ++i) {
                float mean = 0.0f;
                float var = 0.0f;
                
                // Compute mean and variance (cached from forward pass)
                for (size_t j = 0; j < input_cache_.cols(); ++j) {
                    mean += input_cache_(i, j);
                }
                mean /= input_cache_.cols();
                
                for (size_t j = 0; j < input_cache_.cols(); ++j) {
                    float diff = input_cache_(i, j) - mean;
                    var += diff * diff;
                }
                var /= input_cache_.cols();
                
                float std = std::sqrt(var + eps_);
                float inv_std = 1.0f / std;
                
                // Compute gradients
                for (size_t j = 0; j < input_cache_.cols(); ++j) {
                    float x_norm = (input_cache_(i, j) - mean) * inv_std;
                    grad_gamma_(0, j) += grad_output(i, j) * x_norm;
                    grad_beta_(0, j) += grad_output(i, j);
                    
                    // Gradient with respect to input
                    grad_input(i, j) = gamma_(0, j) * grad_output(i, j) * inv_std;
                }
            }
#ifdef USE_CUDA
        }
#endif
        return grad_input;
    } catch (const std::exception& e) {
        throw std::runtime_error("LayerNorm backward failed: " + std::string(e.what()));
    }
}

void LayerNorm::save(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&eps_), sizeof(eps_));
    os.write(reinterpret_cast<const char*>(&hidden_size_), sizeof(hidden_size_));
    // Save gamma and beta as raw data
    os.write(reinterpret_cast<const char*>(gamma_.data()), hidden_size_ * sizeof(float));
    os.write(reinterpret_cast<const char*>(beta_.data()), hidden_size_ * sizeof(float));
}

std::unique_ptr<LayerNorm> LayerNorm::load(std::istream& is) {
    float eps;
    is.read(reinterpret_cast<char*>(&eps), sizeof(eps));

    // Read hidden size from stream
    size_t hidden_size;
    is.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));

    auto ln = std::make_unique<LayerNorm>(hidden_size, eps);

    // Load gamma and beta data directly
    is.read(reinterpret_cast<char*>(ln->gamma_.data()), hidden_size * sizeof(float));
    is.read(reinterpret_cast<char*>(ln->beta_.data()), hidden_size * sizeof(float));

    return ln;
}