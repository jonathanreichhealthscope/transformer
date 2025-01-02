#pragma once

#include <vector>
#include "matrix.hpp"

class Tensor {
public:
    Tensor(unsigned long d1, unsigned long d2, unsigned long d3, unsigned long d4);
    Tensor(const Matrix& mat, const std::vector<unsigned long>& shape);
    
    // Matrix-like interface
    size_t rows() const { return dims_[0] * dims_[1] * dims_[2]; }
    size_t cols() const { return dims_[3]; }
    float& operator()(size_t i, size_t j) { return data_[i * cols() + j]; }
    float operator()(size_t i, size_t j) const { return data_[i * cols() + j]; }
    
    // Tensor-specific access
    float& at(unsigned long i, unsigned long j, unsigned long k, unsigned long l);
    float at(unsigned long i, unsigned long j, unsigned long k, unsigned long l) const;
    
    // Data access
    std::vector<float>& data() { return data_; }
    const std::vector<float>& data() const { return data_; }
    size_t size() const { return data_.size(); }
    
    // Operations
    Tensor transpose(const std::vector<unsigned long>& perm) const;
    Tensor tensormul(const Tensor& other) const;
    Matrix to_matrix() const;
    
    // Conversion operators
    operator Matrix() const { return to_matrix(); }
    
    static Tensor safe_tensormul(const Tensor& a, const Tensor& b);
    
    const std::vector<unsigned long>& dims() const { return dims_; }

private:
    std::vector<unsigned long> dims_;
    std::vector<float> data_;
}; 