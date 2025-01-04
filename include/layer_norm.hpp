#pragma once
#include "components.hpp"
#include <memory>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

class LayerNorm {
private:
    size_t hidden_size;
    float eps;
    Vector gamma;    // Scale parameter
    Vector beta;     // Shift parameter
    Vector mean_cache;
    Vector var_cache;
    Vector norm_cache;
    Matrix normalized;  // Cache for backward pass
    
    // Gradient storage
    Vector gamma_grad;
    Vector beta_grad;
    
    // Store parameter and gradient references
    std::vector<std::reference_wrapper<Vector>> params;
    std::vector<std::reference_wrapper<Vector>> grad_params;

public:
    LayerNorm(size_t hidden_size_, float eps_ = 1e-5) 
        : hidden_size(hidden_size_), eps(eps_),
          gamma(hidden_size, 1.0f), beta(hidden_size, 0.0f),
          mean_cache(hidden_size), var_cache(hidden_size), norm_cache(hidden_size),
          gamma_grad(hidden_size), beta_grad(hidden_size) {
        // Initialize parameter references
        params = {std::ref(gamma), std::ref(beta)};
        grad_params = {std::ref(gamma_grad), std::ref(beta_grad)};
    }

    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& grad_output, const Matrix& input);

    #ifdef USE_CUDA
    Matrix backward_cuda(const Matrix& grad_output, const Matrix& input) const;
    #endif

    // Get parameters for optimization
    std::vector<std::reference_wrapper<Vector>>& parameters() {
        return params;
    }
    
    // Get parameter gradients
    const std::vector<std::reference_wrapper<Vector>>& parameter_gradients() const {
        return grad_params;
    }

    void save(std::ostream& os) const;
    static std::unique_ptr<LayerNorm> load(std::istream& is);

    // Getters for CUDA implementation
    #ifdef USE_CUDA
    const Vector& get_gamma() const { return gamma; }
    const Vector& get_beta() const { return beta; }
    float get_eps() const { return eps; }
    size_t get_hidden_size() const { return hidden_size; }
    #endif

    // Copy constructor
    LayerNorm(const LayerNorm& other)
        : hidden_size(other.hidden_size), eps(other.eps),
          gamma(other.gamma), beta(other.beta),
          mean_cache(other.mean_cache), var_cache(other.var_cache), 
          norm_cache(other.norm_cache),
          normalized(other.normalized),
          gamma_grad(other.gamma_grad), beta_grad(other.beta_grad) {
        // Initialize parameter references
        params = {std::ref(gamma), std::ref(beta)};
        grad_params = {std::ref(gamma_grad), std::ref(beta_grad)};
    }

    // Assignment operator
    LayerNorm& operator=(const LayerNorm& other) {
        if (this != &other) {
            hidden_size = other.hidden_size;
            eps = other.eps;
            gamma = other.gamma;
            beta = other.beta;
            mean_cache = other.mean_cache;
            var_cache = other.var_cache;
            norm_cache = other.norm_cache;
            normalized = other.normalized;
            gamma_grad = other.gamma_grad;
            beta_grad = other.beta_grad;
            // Update parameter references
            params = {std::ref(gamma), std::ref(beta)};
            grad_params = {std::ref(gamma_grad), std::ref(beta_grad)};
        }
        return *this;
    }
}; 