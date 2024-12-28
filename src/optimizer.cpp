#include "optimizer.hpp"
#include <cmath>

AdamOptimizer::AdamOptimizer(float lr, float b1, float b2, float eps)
    : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

void AdamOptimizer::update(const std::vector<Matrix>& params, 
                          const std::vector<Matrix>& grads) {
    // Initialize moment vectors if needed
    if (m.empty()) {
        m.reserve(params.size());
        v.reserve(params.size());
        for (const auto& param : params) {
            m.emplace_back(param.rows(), param.cols(), 0.0f);
            v.emplace_back(param.rows(), param.cols(), 0.0f);
        }
    }
    
    // Increment timestep
    t++;
    
    // Compute bias correction terms
    float bc1 = 1.0f - std::pow(beta1, t);
    float bc2 = 1.0f - std::pow(beta2, t);
    float lr_t = learning_rate * std::sqrt(bc2) / bc1;
    
    // Update each parameter
    for (size_t i = 0; i < params.size(); ++i) {
        const auto& param = params[i];
        const auto& grad = grads[i];
        
        // Update moment estimates
        for (size_t row = 0; row < param.rows(); ++row) {
            for (size_t col = 0; col < param.cols(); ++col) {
                float g = grad(row, col);
                
                // Update biased first moment estimate
                m[i](row, col) = beta1 * m[i](row, col) + (1.0f - beta1) * g;
                
                // Update biased second raw moment estimate
                v[i](row, col) = beta2 * v[i](row, col) + (1.0f - beta2) * g * g;
                
                // Compute bias-corrected moment estimates
                float m_hat = m[i](row, col);
                float v_hat = v[i](row, col);
                
                // Update parameters
                float update = lr_t * m_hat / (std::sqrt(v_hat) + epsilon);
                const_cast<Matrix&>(param)(row, col) -= update;
            }
        }
    }
}

void AdamOptimizer::save(std::ostream& os) const {
    // Save optimizer state
    os.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
    os.write(reinterpret_cast<const char*>(&beta1), sizeof(beta1));
    os.write(reinterpret_cast<const char*>(&beta2), sizeof(beta2));
    os.write(reinterpret_cast<const char*>(&epsilon), sizeof(epsilon));
    os.write(reinterpret_cast<const char*>(&t), sizeof(t));
    
    // Save moment vectors
    size_t size = m.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    
    for (size_t i = 0; i < size; ++i) {
        m[i].save(os);
        v[i].save(os);
    }
}

void AdamOptimizer::load(std::istream& is) {
    // Load optimizer state
    is.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
    is.read(reinterpret_cast<char*>(&beta1), sizeof(beta1));
    is.read(reinterpret_cast<char*>(&beta2), sizeof(beta2));
    is.read(reinterpret_cast<char*>(&epsilon), sizeof(epsilon));
    is.read(reinterpret_cast<char*>(&t), sizeof(t));
    
    // Load moment vectors
    size_t size;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    
    m.clear();
    v.clear();
    m.reserve(size);
    v.reserve(size);
    
    for (size_t i = 0; i < size; ++i) {
        m.push_back(Matrix::load(is));
        v.push_back(Matrix::load(is));
    }
} 