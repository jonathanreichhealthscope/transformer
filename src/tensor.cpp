#include "tensor.hpp"
#include <numeric>
#include <functional>

Tensor::Tensor(size_t dim1, size_t dim2, size_t dim3) 
    : Matrix(dim1 * dim2, dim3), dims{dim1, dim2, dim3} {
}

Tensor::Tensor(size_t dim1, size_t dim2, size_t dim3, size_t dim4) 
    : Matrix(dim1 * dim2, dim3 * dim4), dims{dim1, dim2, dim3, dim4} {
}

Tensor::Tensor(const Matrix& matrix, const std::vector<size_t>& dimensions) 
    : Matrix(matrix), dims(dimensions) {
    // Verify that dimensions match the total size
    size_t total = std::accumulate(dimensions.begin(), dimensions.end(), 
                                 1ULL, std::multiplies<size_t>());
    if (total != matrix.size()) {
        throw std::runtime_error("Dimension mismatch in Tensor construction");
    }
}

size_t Tensor::dim(size_t index) const {
    if (index >= dims.size()) {
        throw std::out_of_range("Dimension index out of range");
    }
    return dims[index];
}

void Tensor::reshape(const std::vector<size_t>& new_dims) {
    size_t total = std::accumulate(new_dims.begin(), new_dims.end(), 
                                 1ULL, std::multiplies<size_t>());
    if (total != size()) {
        throw std::runtime_error("Invalid reshape dimensions");
    }
    dims = new_dims;
}

float& Tensor::at(size_t i, size_t j, size_t k) {
    if (dims.size() != 3) {
        throw std::runtime_error("Tensor is not 3-dimensional");
    }
    size_t index = (i * dims[1] + j) * dims[2] + k;
    return data()[index];
}

float& Tensor::at(size_t i, size_t j, size_t k, size_t l) {
    if (dims.size() != 4) {
        throw std::runtime_error("Tensor is not 4-dimensional");
    }
    size_t index = ((i * dims[1] + j) * dims[2] + k) * dims[3] + l;
    return data()[index];
}

const float& Tensor::at(size_t i, size_t j, size_t k) const {
    if (dims.size() != 3) {
        throw std::runtime_error("Tensor is not 3-dimensional");
    }
    size_t index = (i * dims[1] + j) * dims[2] + k;
    return data()[index];
}

const float& Tensor::at(size_t i, size_t j, size_t k, size_t l) const {
    if (dims.size() != 4) {
        throw std::runtime_error("Tensor is not 4-dimensional");
    }
    size_t index = ((i * dims[1] + j) * dims[2] + k) * dims[3] + l;
    return data()[index];
}

Matrix Tensor::to_matrix() const {
    // For 3D tensor: collapse first two dimensions
    if (dims.size() == 3) {
        return Matrix(dims[0] * dims[1], dims[2], data());
    }
    // For 4D tensor: collapse first two and last two dimensions
    else if (dims.size() == 4) {
        return Matrix(dims[0] * dims[1], dims[2] * dims[3], data());
    }
    throw std::runtime_error("Unsupported tensor rank for matrix conversion");
}

Tensor Tensor::from_matrix(const Matrix& matrix, const std::vector<size_t>& dims) {
    return Tensor(matrix, dims);
}

Tensor Tensor::transpose(const std::vector<size_t>& axes) const {
    if (axes.size() != dims.size()) {
        throw std::runtime_error("Invalid axes for transpose");
    }
    
    // Create new dimensions based on the axes
    std::vector<size_t> new_dims;
    new_dims.reserve(dims.size());
    for (size_t axis : axes) {
        new_dims.push_back(dims[axis]);
    }
    
    // Create new tensor with transposed data
    Tensor result(new_dims[0], new_dims[1], new_dims[2]);
    
    // Implement the transpose operation
    // This is a basic implementation for 3D tensors
    if (dims.size() == 3) {
        for (size_t i = 0; i < dims[0]; ++i) {
            for (size_t j = 0; j < dims[1]; ++j) {
                for (size_t k = 0; k < dims[2]; ++k) {
                    std::vector<size_t> old_indices = {i, j, k};
                    std::vector<size_t> new_indices(3);
                    for (size_t ax = 0; ax < 3; ++ax) {
                        new_indices[ax] = old_indices[axes[ax]];
                    }
                    result.at(new_indices[0], new_indices[1], new_indices[2]) = 
                        at(i, j, k);
                }
            }
        }
    }
    
    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    // Implement tensor multiplication
    // This is a basic implementation for batched matrix multiplication
    if (dims.size() != other.dims.size()) {
        throw std::runtime_error("Tensor ranks must match for multiplication");
    }
    
    if (dims.size() == 3) {
        if (dims[2] != other.dims[1]) {
            throw std::runtime_error("Invalid dimensions for tensor multiplication");
        }
        
        Tensor result(dims[0], dims[1], other.dims[2]);
        
        // Perform batched matrix multiplication
        for (size_t b = 0; b < dims[0]; ++b) {
            Matrix m1(dims[1], dims[2]);
            Matrix m2(other.dims[1], other.dims[2]);
            
            // Extract matrices for this batch
            for (size_t i = 0; i < dims[1]; ++i) {
                for (size_t j = 0; j < dims[2]; ++j) {
                    m1(i, j) = at(b, i, j);
                }
            }
            
            for (size_t i = 0; i < other.dims[1]; ++i) {
                for (size_t j = 0; j < other.dims[2]; ++j) {
                    m2(i, j) = other.at(b, i, j);
                }
            }
            
            // Multiply matrices
            Matrix res = matmul(m1, m2);
            
            // Store result
            for (size_t i = 0; i < res.rows(); ++i) {
                for (size_t j = 0; j < res.cols(); ++j) {
                    result.at(b, i, j) = res(i, j);
                }
            }
        }
        
        return result;
    }
    
    throw std::runtime_error("Unsupported tensor rank for multiplication");
} 