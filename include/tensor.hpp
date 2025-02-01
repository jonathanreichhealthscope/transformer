#pragma once

#include "matrix.hpp"
#include <vector>
#include <string>

/**
 * @brief A 4-dimensional tensor class for neural network computations.
 * 
 * The Tensor class provides a 4D data structure commonly used in deep learning,
 * particularly for attention mechanisms and batch processing. It supports:
 * - Matrix-like 2D operations
 * - 4D tensor operations
 * - Dimension permutation and transposition
 * - Tensor multiplication
 * - Conversion to/from matrices
 */
class Tensor {
  public:
    /**
     * @brief Constructs a 4D tensor with specified dimensions.
     * @param d1 First dimension size (typically batch size)
     * @param d2 Second dimension size (typically number of heads)
     * @param d3 Third dimension size (typically sequence length)
     * @param d4 Fourth dimension size (typically head dimension)
     */
    Tensor(unsigned long d1, unsigned long d2, unsigned long d3, unsigned long d4);

    /**
     * @brief Constructs a tensor from a matrix with specified shape.
     * @param mat Source matrix
     * @param shape Target tensor dimensions
     */
    Tensor(const Matrix& mat, const std::vector<unsigned long>& shape);

    /**
     * @brief Gets the number of rows in the 2D view of the tensor.
     * @return Product of first three dimensions
     */
    size_t rows() const {
        return dims_[0] * dims_[1] * dims_[2];
    }

    /**
     * @brief Gets the number of columns in the 2D view of the tensor.
     * @return Size of the fourth dimension
     */
    size_t cols() const {
        return dims_[3];
    }

    /**
     * @brief Accesses tensor element in 2D view (mutable).
     * @param i Row index
     * @param j Column index
     * @return Reference to the element
     */
    float& operator()(size_t i, size_t j) {
        return data_[i * cols() + j];
    }

    /**
     * @brief Accesses tensor element in 2D view (const).
     * @param i Row index
     * @param j Column index
     * @return Value of the element
     */
    float operator()(size_t i, size_t j) const {
        return data_[i * cols() + j];
    }

    /**
     * @brief Accesses tensor element in 4D view (mutable).
     * @param i First dimension index
     * @param j Second dimension index
     * @param k Third dimension index
     * @param l Fourth dimension index
     * @return Reference to the element
     */
    float& at(unsigned long i, unsigned long j, unsigned long k, unsigned long l);

    /**
     * @brief Accesses tensor element in 4D view (const).
     * @param i First dimension index
     * @param j Second dimension index
     * @param k Third dimension index
     * @param l Fourth dimension index
     * @return Value of the element
     */
    float at(unsigned long i, unsigned long j, unsigned long k, unsigned long l) const;

    /**
     * @brief Gets the underlying data vector (mutable).
     * @return Reference to the data vector
     */
    std::vector<float>& data() {
        return data_;
    }

    /**
     * @brief Gets the underlying data vector (const).
     * @return Const reference to the data vector
     */
    const std::vector<float>& data() const {
        return data_;
    }

    /**
     * @brief Gets the total number of elements in the tensor.
     * @return Product of all dimensions
     */
    size_t size() const {
        return data_.size();
    }

    /**
     * @brief Transposes the tensor according to the given permutation.
     * @param perm Vector specifying the new order of dimensions
     * @return Transposed tensor
     */
    Tensor transpose(const std::vector<unsigned long>& perm) const;

    /**
     * @brief Permutes the tensor dimensions.
     * @param perm Vector specifying the new order of dimensions
     * @return Permuted tensor
     */
    Tensor permute(const std::vector<unsigned long>& perm) const;

    /**
     * @brief Performs tensor multiplication.
     * @param other Tensor to multiply with
     * @return Result of tensor multiplication
     */
    Tensor tensormul(const Tensor& other) const;

    /**
     * @brief Converts the tensor to a matrix.
     * @return Matrix representation of the tensor
     */
    Matrix to_matrix() const;

    /**
     * @brief Creates a new tensor filled with a constant value.
     * @param value Value to fill with
     * @return Filled tensor
     */
    Tensor fill(float value) const;

    /**
     * @brief Implicit conversion to Matrix.
     */
    operator Matrix() const {
        return to_matrix();
    }

    /**
     * @brief Performs safe tensor multiplication with dimension checking.
     * @param a First tensor
     * @param b Second tensor
     * @return Result of tensor multiplication
     * @throws std::runtime_error if dimensions are incompatible
     */
    static Tensor safe_tensormul(const Tensor& a, const Tensor& b);

    /**
     * @brief Gets the tensor dimensions.
     * @return Vector of dimension sizes
     */
    const std::vector<unsigned long>& dims() const {
        return dims_;
    }

    /**
     * @brief Validates matrix dimensions for operations
     * @param other Other tensor to validate against
     * @param operation Name of operation being validated ("multiplication" or "addition")
     * @throws std::runtime_error if dimensions are invalid for the operation
     */
    void validate_matrix_dimensions(const Tensor& other, const std::string& operation) const;

    /**
     * @brief Applies softmax operation along the last dimension.
     */
    void softmax();

  private:
    std::vector<unsigned long> dims_;  ///< Sizes of each dimension
    std::vector<float> data_;         ///< Flattened tensor data
};