#include "../include/optimizer.hpp"

Optimizer::Optimizer(float lr, float b1, float b2, float eps)
    : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

void Optimizer::add_parameter(Matrix& param) {
    parameters.push_back(&param);
    gradients.emplace_back(param.rows(), param.cols(), 0.0f);
}

void Optimizer::update(const std::vector<Matrix>& params, const std::vector<Matrix>& grads) {
    if (params.size() != grads.size()) {
        throw std::runtime_error("Parameter and gradient count mismatch");
    }

    for (size_t i = 0; i < params.size(); ++i) {
        if (i >= parameters.size()) {
            add_parameter(const_cast<Matrix&>(params[i]));
        }
        gradients[i] = grads[i];
    }
}

void Optimizer::step(std::vector<Matrix*>& params, const std::vector<Matrix>& grads) {
    // Convert vector of pointers to vector of references
    std::vector<Matrix> param_refs;
    param_refs.reserve(params.size());
    for (auto* param : params) {
        param_refs.push_back(*param);
    }
    update(param_refs, grads);
    t++;  // Increment timestep
    
    for (size_t i = 0; i < parameters.size(); ++i) {
        Matrix& param = *parameters[i];
        const Matrix& grad = gradients[i];
        
        // Apply Adam-style update
        for (size_t r = 0; r < param.rows(); ++r) {
            for (size_t c = 0; c < param.cols(); ++c) {
                float g = grad(r, c);
                
                // Bias correction
                float lr_t = learning_rate * std::sqrt(1.0f - std::pow(beta2, t)) / 
                           (1.0f - std::pow(beta1, t));
                
                // Update parameter
                param(r, c) -= lr_t * g;
            }
        }
    }
}

void Optimizer::zero_grad() {
    for (auto& grad : gradients) {
        grad.fill(0.0f);
    }
}

void Optimizer::save(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
    os.write(reinterpret_cast<const char*>(&beta1), sizeof(beta1));
    os.write(reinterpret_cast<const char*>(&beta2), sizeof(beta2));
    os.write(reinterpret_cast<const char*>(&epsilon), sizeof(epsilon));
    os.write(reinterpret_cast<const char*>(&t), sizeof(t));
}

void Optimizer::load(std::istream& is) {
    is.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
    is.read(reinterpret_cast<char*>(&beta1), sizeof(beta1));
    is.read(reinterpret_cast<char*>(&beta2), sizeof(beta2));
    is.read(reinterpret_cast<char*>(&epsilon), sizeof(epsilon));
    is.read(reinterpret_cast<char*>(&t), sizeof(t));
} 