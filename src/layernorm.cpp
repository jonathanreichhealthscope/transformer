#include "../include/layernorm.hpp"
#include <cmath>

LayerNorm::LayerNorm(size_t hidden_size, float eps)
    : gamma(hidden_size, 1.0f), beta(hidden_size, 0.0f), eps(eps) {}

Matrix LayerNorm::forward(const Matrix& x) const {
    const size_t batch_size = x.rows();
    const size_t hidden_size = x.cols();
    Matrix result(batch_size, hidden_size);
    
    for (size_t i = 0; i < batch_size; ++i) {
        // Compute mean
        float mean = 0.0f;
        for (size_t j = 0; j < hidden_size; ++j) {
            mean += x(i, j);
        }
        mean /= hidden_size;
        
        // Compute variance
        float var = 0.0f;
        for (size_t j = 0; j < hidden_size; ++j) {
            float diff = x(i, j) - mean;
            var += diff * diff;
        }
        var /= hidden_size;
        
        // Normalize and apply scale/shift
        float std = std::sqrt(var + eps);
        for (size_t j = 0; j < hidden_size; ++j) {
            result(i, j) = gamma[j] * ((x(i, j) - mean) / std) + beta[j];
        }
    }
    
    return result;
}

Matrix LayerNorm::forward_cuda(const Matrix& x) const {
#ifdef USE_CUDA
    // Implementation in cuda/layernorm_kernels.cu
    throw std::runtime_error("CUDA implementation not available");
#else
    return forward(x);
#endif
}

void LayerNorm::save(std::ostream& os) const {
    gamma.save(os);
    beta.save(os);
    os.write(reinterpret_cast<const char*>(&eps), sizeof(eps));
}

std::unique_ptr<LayerNorm> LayerNorm::load(std::istream& is) {
    Vector gamma = Vector::load(is);
    Vector beta = Vector::load(is);
    float eps;
    is.read(reinterpret_cast<char*>(&eps), sizeof(eps));
    
    auto ln = std::make_unique<LayerNorm>(gamma.size(), eps);
    ln->gamma = std::move(gamma);
    ln->beta = std::move(beta);
    return ln;
} 