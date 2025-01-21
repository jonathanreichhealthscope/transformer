#include "../include/layer_norm.hpp"
#include <cmath>
#include <omp.h>

Matrix LayerNorm::forward(const Matrix& x) {
    const float eps = 1e-5f; // Increased epsilon for better stability

    size_t batch_size = x.rows();
    size_t hidden_size = x.cols();
    Matrix output(batch_size, hidden_size);
    normalized = Matrix(batch_size, hidden_size); // Store normalized values

#pragma omp parallel for
    for (size_t i = 0; i < batch_size; i++) {
        // Compute mean with Kahan summation for numerical stability
        double mean = 0.0;
        double c = 0.0; // Compensation term
        for (size_t j = 0; j < hidden_size; j++) {
            double y = x(i, j) - c;
            double t = mean + y;
            c = (t - mean) - y;
            mean = t;
        }
        mean /= hidden_size;

        // Compute variance with two-pass algorithm for stability
        double var = 0.0;
        c = 0.0;
        for (size_t j = 0; j < hidden_size; j++) {
            double diff = x(i, j) - mean;
            double y = diff * diff - c;
            double t = var + y;
            c = (t - var) - y;
            var = t;
        }
        var = var / hidden_size;

        // Normalize with bounds checking
        float std_dev = std::sqrt(var + eps);
        if (std_dev < eps) {
            std_dev = eps; // Prevent division by very small numbers
        }

        for (size_t j = 0; j < hidden_size; j++) {
            float norm_val = (x(i, j) - mean) / std_dev;
            // Clip extreme values
            norm_val = std::max(std::min(norm_val, 1e4f), -1e4f);
            normalized(i, j) = norm_val; // Store normalized value
            output(i, j) = norm_val * gamma[j] + beta[j];
        }
    }

    return output;
}

Matrix LayerNorm::backward(const Matrix& grad, const Matrix& input) {
    Matrix dx(grad.rows(), grad.cols());

    // Reset gradients
    std::fill(gamma_grad.data(), gamma_grad.data() + gamma_grad.size(), 0.0f);
    std::fill(beta_grad.data(), beta_grad.data() + beta_grad.size(), 0.0f);

// Compute gradients with respect to normalized inputs
#pragma omp parallel for
    for (size_t i = 0; i < grad.size(); i++) {
        dx.data()[i] = grad.data()[i] * gamma[i % hidden_size];
    }

    // Compute gradients for gamma and beta
    for (size_t i = 0; i < grad.size(); i++) {
        size_t param_idx = i % hidden_size;
        gamma_grad[param_idx] += grad.data()[i] * normalized.data()[i];
        beta_grad[param_idx] += grad.data()[i];
    }

    return dx;
}

void LayerNorm::save(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&eps), sizeof(eps));
    gamma.save(os);
    beta.save(os);
}

std::unique_ptr<LayerNorm> LayerNorm::load(std::istream& is) {
    float eps;
    is.read(reinterpret_cast<char*>(&eps), sizeof(eps));

    Vector gamma_vec = Vector::load(is);
    Vector beta_vec = Vector::load(is);

    auto ln = std::make_unique<LayerNorm>(gamma_vec.size(), eps);
    ln->gamma = std::move(gamma_vec);
    ln->beta = std::move(beta_vec);
    return ln;
}