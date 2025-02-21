#include "../include/tensor.hpp"
#include <numeric>
#include <omp.h>
#include <stdexcept>

Tensor::Tensor(unsigned long d1, unsigned long d2, unsigned long d3, unsigned long d4)
    : dims_{d1, d2, d3, d4} {
    size_t total_size = d1 * d2 * d3 * d4;
    data_.resize(total_size, 0.0f);
}

Tensor::Tensor(const Matrix& mat, const std::vector<unsigned long>& shape) {
    if (shape.size() != 4) {
        throw std::runtime_error("Tensor shape must have 4 dimensions");
    }

    size_t total_size =
        std::accumulate(shape.begin(), shape.end(), 1UL, std::multiplies<unsigned long>());
    if (total_size != mat.size()) {
        throw std::runtime_error("Matrix size does not match tensor shape");
    }

    dims_ = shape;
    data_ = std::vector<float>(mat.data(), mat.data() + mat.size());
}

float& Tensor::at(unsigned long i, unsigned long j, unsigned long k, unsigned long l) {
    size_t index =
        i * (dims_[1] * dims_[2] * dims_[3]) + j * (dims_[2] * dims_[3]) + k * dims_[3] + l;
    if (index >= data_.size()) {
        throw std::out_of_range("Tensor index out of bounds");
    }
    return data_[index];
}

float Tensor::at(unsigned long i, unsigned long j, unsigned long k, unsigned long l) const {
    size_t index =
        i * (dims_[1] * dims_[2] * dims_[3]) + j * (dims_[2] * dims_[3]) + k * dims_[3] + l;
    if (index >= data_.size()) {
        throw std::out_of_range("Tensor index out of bounds");
    }
    return data_[index];
}
Tensor Tensor::fill(float value) const {
    Tensor result(dims_[0], dims_[1], dims_[2], dims_[3]);
    std::fill(result.data_.begin(), result.data_.end(), value);
    return result;
}

Tensor Tensor::transpose(const std::vector<unsigned long>& perm) const {
    if (perm.size() != 4) {
        throw std::runtime_error("Transpose permutation must have 4 dimensions");
    }

    std::vector<unsigned long> new_dims = {dims_[perm[0]], dims_[perm[1]], dims_[perm[2]],
                                           dims_[perm[3]]};

    Tensor result(new_dims[0], new_dims[1], new_dims[2], new_dims[3]);

    for (size_t i = 0; i < dims_[0]; ++i) {
        for (size_t j = 0; j < dims_[1]; ++j) {
            for (size_t k = 0; k < dims_[2]; ++k) {
                for (size_t l = 0; l < dims_[3]; ++l) {
                    std::vector<size_t> old_idx = {i, j, k, l};
                    std::vector<size_t> new_idx = {old_idx[perm[0]], old_idx[perm[1]],
                                                   old_idx[perm[2]], old_idx[perm[3]]};
                    result.at(new_idx[0], new_idx[1], new_idx[2], new_idx[3]) = at(i, j, k, l);
                }
            }
        }
    }

    return result;
}

Tensor Tensor::permute(const std::vector<unsigned long>& perm) const {
    if (perm.size() != 4) {
        throw std::runtime_error("Permutation must be of size 4");
    }

    std::vector<unsigned long> new_dims = {dims_[perm[0]], dims_[perm[1]], dims_[perm[2]],
                                           dims_[perm[3]]};

    Tensor result(new_dims[0], new_dims[1], new_dims[2], new_dims[3]);

#pragma omp parallel for collapse(4)
    for (size_t i = 0; i < dims_[0]; ++i) {
        for (size_t j = 0; j < dims_[1]; ++j) {
            for (size_t k = 0; k < dims_[2]; ++k) {
                for (size_t l = 0; l < dims_[3]; ++l) {
                    std::vector<size_t> old_idx = {i, j, k, l};
                    std::vector<size_t> new_idx = {old_idx[perm[0]], old_idx[perm[1]],
                                                   old_idx[perm[2]], old_idx[perm[3]]};
                    result.at(new_idx[0], new_idx[1], new_idx[2], new_idx[3]) = at(i, j, k, l);
                }
            }
        }
    }

    return result;
}

Tensor Tensor::tensormul(const Tensor& other) const {
    if (dims_[3] != other.dims_[2]) {
        throw std::runtime_error("Incompatible dimensions for tensor multiplication");
    }

    Tensor result(dims_[0], dims_[1], dims_[2], other.dims_[3]);

#pragma omp parallel for collapse(4)
    for (size_t i = 0; i < dims_[0]; ++i) {
        for (size_t j = 0; j < dims_[1]; ++j) {
            for (size_t k = 0; k < dims_[2]; ++k) {
                for (size_t l = 0; l < other.dims_[3]; ++l) {
                    float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
                    for (size_t m = 0; m < dims_[3]; ++m) {
                        sum += at(i, j, k, m) * other.at(i, j, m, l);
                    }
                    result.at(i, j, k, l) = sum;
                }
            }
        }
    }

    return result;
}

Matrix Tensor::to_matrix() const {
    size_t rows = this->rows();
    size_t cols = this->cols();

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data_[i * cols + j];
        }
    }

    return result;
}

Tensor Tensor::safe_tensormul(const Tensor& a, const Tensor& b) {
    try {
        return a.tensormul(b);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Safe tensor multiplication failed: ") + e.what());
    }
}

void Tensor::validate_matrix_dimensions(const Tensor& other, const std::string& operation) const {
    if (rows() <= 0 || cols() <= 0 || other.rows() <= 0 || other.cols() <= 0) {
        throw std::runtime_error("Matrix dimensions cannot be zero for " + operation);
    }
    
    if (operation == "multiplication") {
        if (cols() != other.rows()) {
            throw std::runtime_error("Invalid matrix dimensions for multiplication: " + 
                std::to_string(rows()) + "x" + std::to_string(cols()) + " * " +
                std::to_string(other.rows()) + "x" + std::to_string(other.cols()));
        }
    } else if (operation == "addition") {
        if (rows() != other.rows() || cols() != other.cols()) {
            throw std::runtime_error("Invalid matrix dimensions for addition: " +
                std::to_string(rows()) + "x" + std::to_string(cols()) + " + " +
                std::to_string(other.rows()) + "x" + std::to_string(other.cols()));
        }
    }
}

void Tensor::softmax() {
    // Find max for each row separately for numerical stability
    for (size_t i = 0; i < rows(); i++) {
        // Find max in this row
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < cols(); j++) {
            max_val = std::max(max_val, operator()(i, j));
        }
        
        // Compute exp(x - max) for numerical stability
        float sum_exp = 0.0f;
        for (size_t j = 0; j < cols(); j++) {
            float val = std::exp(operator()(i, j) - max_val);
            operator()(i, j) = val;
            sum_exp += val;
        }
        
        // Normalize to get probabilities
        if (sum_exp > 0.0f) {  // Protect against division by zero
            for (size_t j = 0; j < cols(); j++) {
                operator()(i, j) /= sum_exp;
            }
        }
    }
}
