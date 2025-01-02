#include "tensor.hpp"
#include <numeric>
#include <functional>
#include <string>
Tensor::Tensor(size_t dim1, size_t dim2, size_t dim3) 
    : Matrix(dim1 * dim2, dim3), dims_{dim1, dim2, dim3} {
}

Tensor::Tensor(size_t dim1, size_t dim2, size_t dim3, size_t dim4) 
    : Matrix(dim1 * dim2, dim3 * dim4), dims_{dim1, dim2, dim3, dim4} {
}

Tensor::Tensor(const Matrix& matrix, const std::vector<size_t>& dimensions) 
    : Matrix(matrix), dims_(dimensions) {
    // Verify that dimensions match the total size
    size_t total = std::accumulate(dimensions.begin(), dimensions.end(), 
                                 1ULL, std::multiplies<size_t>());
    if (total != matrix.size()) {
        throw std::runtime_error("Dimension mismatch in Tensor construction");
    }
}

size_t Tensor::dim(size_t index) const {
    if (index >= dims_.size()) {
        throw std::out_of_range("Dimension index out of range");
    }
    return dims_[index];
}

void Tensor::reshape(const std::vector<size_t>& new__dims) {
    size_t total = std::accumulate(new__dims.begin(), new__dims.end(), 
                                 1ULL, std::multiplies<size_t>());
    if (total != size()) {
        throw std::runtime_error("Invalid reshape dimensions");
    }
    dims_ = new__dims;
}

float& Tensor::at(size_t i, size_t j, size_t k) {
    if (dims_.size() != 3) {
        throw std::runtime_error("Tensor is not 3-dimensional");
    }
    size_t index = (i * dims_[1] + j) * dims_[2] + k;
    return data()[index];
}

float& Tensor::at(size_t i, size_t j, size_t k, size_t l) {
    if (dims_.size() != 4) {
        throw std::runtime_error("Tensor is not 4-dimensional");
    }
    size_t index = ((i * dims_[1] + j) * dims_[2] + k) * dims_[3] + l;
    return data()[index];
}

const float& Tensor::at(size_t i, size_t j, size_t k) const {
    if (dims_.size() != 3) {
        throw std::runtime_error("Tensor is not 3-dimensional");
    }
    size_t index = (i * dims_[1] + j) * dims_[2] + k;
    return data()[index];
}

const float& Tensor::at(size_t i, size_t j, size_t k, size_t l) const {
    if (dims_.size() != 4) {
        throw std::runtime_error("Tensor is not 4-dimensional");
    }
    size_t index = ((i * dims_[1] + j) * dims_[2] + k) * dims_[3] + l;
    return data()[index];
}

Matrix Tensor::to_matrix() const {
    // For 3D tensor: collapse first two dimensions
    if (dims_.size() == 3) {
        return Matrix(dims_[0] * dims_[1], dims_[2], data());
    }
    // For 4D tensor: collapse first two and last two dimensions
    else if (dims_.size() == 4) {
        return Matrix(dims_[0] * dims_[1], dims_[2] * dims_[3], data());
    }
    throw std::runtime_error("Unsupported tensor rank for matrix conversion");
}

Tensor Tensor::from_matrix(const Matrix& matrix, const std::vector<size_t>& dims_) {
    return Tensor(matrix, dims_);
}

Tensor Tensor::transpose(const std::vector<size_t>& axes) const {
    // Validate axes
    if (axes.size() != dims_.size()) {
        throw std::runtime_error("Number of axes must match tensor rank");
    }
    
    // Check if axes is a valid permutation
    std::vector<bool> used(dims_.size(), false);
    for (size_t axis : axes) {
        if (axis >= dims_.size() || used[axis]) {
            throw std::runtime_error("Invalid axes permutation");
        }
        used[axis] = true;
    }
    
    // Create new dimensions array based on the permutation
    std::vector<size_t> new__dims(dims_.size());
    for (size_t i = 0; i < dims_.size(); ++i) {
        new__dims[i] = dims_[axes[i]];
    }
    
    // Create new tensor with transposed dimensions
    Tensor result(new__dims[0], new__dims[1], new__dims[2], new__dims[3]);
    
    // Copy data with transposed indices
    for (size_t i = 0; i < size(); ++i) {
        std::vector<size_t> indices = unflatten_index(i);
        size_t transposed_idx = compute_transposed_index(indices, axes);
        result.data()[transposed_idx] = data()[i];
    }
    
    return result;
}

Tensor Tensor::tensormul(const Tensor& other) const {
    // Implement tensor multiplication
    // This is a basic implementation for batched matrix multiplication
    if (dims_.size() != other.dims_.size()) {
        throw std::runtime_error("Tensor ranks must match for multiplication");
    }
    
    if (dims_.size() == 3) {
        if (dims_[2] != other.dims_[1]) {
            throw std::runtime_error("Invalid dimensions for tensor multiplication");
        }
        
        Tensor result(dims_[0], dims_[1], other.dims_[2]);
        
        // Perform batched matrix multiplication
        for (size_t b = 0; b < dims_[0]; ++b) {
            Matrix m1(dims_[1], dims_[2]);
            Matrix m2(other.dims_[1], other.dims_[2]);
            
            // Extract matrices for this batch
            for (size_t i = 0; i < dims_[1]; ++i) {
                for (size_t j = 0; j < dims_[2]; ++j) {
                    m1(i, j) = at(b, i, j);
                }
            }
            
            for (size_t i = 0; i < other.dims_[1]; ++i) {
                for (size_t j = 0; j < other.dims_[2]; ++j) {
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

std::vector<size_t> Tensor::unflatten_index(size_t flat_idx) const {
    std::vector<size_t> indices(dims_.size());
    for (int i = dims_.size() - 1; i >= 0; --i) {
        indices[i] = flat_idx % dims_[i];
        flat_idx /= dims_[i];
    }
    return indices;
}

size_t Tensor::flatten_index(const std::vector<size_t>& indices) const {
    if (indices.size() != dims_.size()) {
        throw std::runtime_error("Invalid number of indices");
    }
    
    size_t flat_idx = 0;
    size_t multiplier = 1;
    
    for (int i = dims_.size() - 1; i >= 0; --i) {
        if (indices[i] >= dims_[i]) {
            throw std::runtime_error("Index out of bounds");
        }
        flat_idx += indices[i] * multiplier;
        multiplier *= dims_[i];
    }
    return flat_idx;
}

size_t Tensor::compute_transposed_index(const std::vector<size_t>& indices,
                                      const std::vector<size_t>& axes) const {
    std::vector<size_t> transposed_indices(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        transposed_indices[i] = indices[axes[i]];
    }
    return flatten_index(transposed_indices);
}

Tensor Tensor::safe_tensormul(const Tensor& A, const Tensor& B) {
    // Validate tensor ranks match
    if (A.rank() != B.rank()) {
        throw std::runtime_error("Tensor ranks must match for multiplication: " +
                               std::to_string(A.rank()) + " != " + 
                               std::to_string(B.rank()));
    }
    
    // For 4D tensors [batch, heads, seq, hidden]
    if (A.rank() == 4) {
        // Check batch and heads dimensions match
        if (A.dim(0) != B.dim(0) || A.dim(1) != B.dim(1)) {
            throw std::runtime_error("Batch or heads dimensions mismatch");
        }
        
        // Check multiplication compatibility
        if (A.dim(3) != B.dim(2)) {
            throw std::runtime_error("Inner dimensions mismatch: " +
                                   std::to_string(A.dim(3)) + " != " + 
                                   std::to_string(B.dim(2)));
        }
        
        return A.matmul(B);
    }
    
    throw std::runtime_error("Unsupported tensor rank for safe_tensormul");
} 