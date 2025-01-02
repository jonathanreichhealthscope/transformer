#pragma once
#include "matrix.hpp"
#include <vector>
#include <stdexcept>

class Tensor : public Matrix {
private:
    std::vector<size_t> dims_;  // Stores dimensions (e.g., [batch, heads, seq_len, hidden])
    
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
    const std::vector<size_t>& get_dims() const { return dims_; }
    size_t rank() const { return dims_.size(); }
    
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
    Tensor tensormul(const Tensor& other) const;
    Tensor matmul(const Matrix& first, const Matrix& second) const;
    
    // Safe tensor multiplication with dimension checks
    static Tensor safe_tensormul(const Tensor& A, const Tensor& B);
    
    // Helper method to compute transposed index
    size_t compute_transposed_index(const std::vector<size_t>& indices,
                                  const std::vector<size_t>& axes) const;
    
    // Helper method to convert flat index to multi-dimensional indices
    std::vector<size_t> unflatten_index(size_t flat_idx) const;
    
    // Helper method to convert multi-dimensional indices to flat index
    size_t flatten_index(const std::vector<size_t>& indices) const;
}; 