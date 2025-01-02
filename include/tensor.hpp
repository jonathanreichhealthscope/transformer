#pragma once
#include "matrix.hpp"
#include <vector>
#include <stdexcept>

class Tensor : public Matrix {
private:
    std::vector<size_t> dims;  // Stores dimensions (e.g., [batch, heads, seq_len, hidden])
    
public:
    // Constructors
    Tensor() = default;
    
    // Constructor for 3D tensor
    Tensor(size_t dim1, size_t dim2, size_t dim3);
    
    // Constructor for 4D tensor
    Tensor(size_t dim1, size_t dim2, size_t dim3, size_t dim4);
    
    // Constructor from Matrix with reshape
    explicit Tensor(const Matrix& matrix, const std::vector<size_t>& dimensions);
    
    // Dimension access
    size_t dim(size_t index) const;
    const std::vector<size_t>& dims() const { return dims; }
    size_t rank() const { return dims.size(); }
    
    // Reshape methods
    void reshape(const std::vector<size_t>& new_dims);
    
    // Access methods (for 3D and 4D)
    float& at(size_t i, size_t j, size_t k);
    float& at(size_t i, size_t j, size_t k, size_t l);
    const float& at(size_t i, size_t j, size_t k) const;
    const float& at(size_t i, size_t j, size_t k, size_t l) const;
    
    // Conversion methods
    Matrix to_matrix() const;  // Flattens to 2D
    static Tensor from_matrix(const Matrix& matrix, const std::vector<size_t>& dims);
    
    // Basic operations
    Tensor transpose(const std::vector<size_t>& axes) const;
    Tensor matmul(const Tensor& other) const;
}; 