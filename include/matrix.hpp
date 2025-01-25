#pragma once
#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>
#include "vector.hpp"  // Include Vector definition from vector.hpp
#define M_PI 3.14159265358979323846
// Forward declarations
class Matrix;
class Vector;

/**
 * @brief A 2D matrix class optimized for neural network operations.
 * 
 * The Matrix class provides a fundamental building block for neural network computations,
 * supporting both CPU and GPU operations. Features include:
 * - Basic matrix operations (addition, multiplication, transposition)
 * - Neural network specific operations (ReLU, GELU, Softmax)
 * - CUDA acceleration support
 * - Memory management for both CPU and GPU
 * - Efficient data access patterns
 */
class Matrix {
  private:
    std::vector<float> data_;        ///< Matrix data storage on CPU
    size_t rows_;                    ///< Number of rows
    size_t cols_;                    ///< Number of columns
    std::tuple<size_t, size_t> shape_; ///< Matrix shape as (rows, cols)
    bool owns_data_ = true;          ///< Whether this matrix owns its data or views external data

#ifdef CUDA_AVAILABLE
    float* gpu_data_ = nullptr;      ///< Matrix data storage on GPU
    bool is_on_gpu_ = false;         ///< Whether the data is currently on GPU
#endif

  public:
    /**
     * @brief Default constructor.
     */
    Matrix();

    /**
     * @brief Constructs a matrix with specified dimensions.
     * @param rows Number of rows
     * @param cols Number of columns
     * @param init_val Initial value for all elements (default: 0.0f)
     */
    Matrix(size_t rows, size_t cols, float init_val = 0.0f);

    /**
     * @brief Constructs a matrix using external data.
     * @param rows Number of rows
     * @param cols Number of columns
     * @param external_data Pointer to external data
     */
    Matrix(size_t rows, size_t cols, float* external_data);

    /**
     * @brief Constructs a matrix using external data with ownership control.
     * @param rows Number of rows
     * @param cols Number of columns
     * @param external_data Pointer to external data
     * @param is_owner Whether this matrix should own the data
     */
    Matrix(size_t rows, size_t cols, float* external_data, bool is_owner);

    /**
     * @brief Gets the number of rows.
     * @return Number of rows
     */
    size_t rows() const {
        return rows_;
    }

    /**
     * @brief Gets the number of columns.
     * @return Number of columns
     */
    size_t cols() const {
        return cols_;
    }

    /**
     * @brief Gets the total number of elements.
     * @return Number of elements
     */
    size_t size() const {
        return data_.size();
    }

    /**
     * @brief Gets the total size in bytes.
     * @return Size in bytes
     */
    size_t bytes() const {
        return size() * sizeof(float);
    }

    /**
     * @brief Gets the matrix shape.
     * @return Tuple of (rows, cols)
     */
    std::tuple<size_t, size_t> shape() const {
        return shape_;
    }

    /**
     * @brief Checks if the matrix is empty.
     * @return True if empty
     */
    bool empty() const {
        return data_.empty();
    }

    /**
     * @brief Gets the minimum value in the matrix.
     * @return Minimum value
     */
    float min() const {
        return *std::min_element(data_.begin(), data_.end());
    }

    /**
     * @brief Gets the maximum value in the matrix.
     * @return Maximum value
     */
    float max() const {
        return *std::max_element(data_.begin(), data_.end());
    }

    /**
     * @brief Gets a pointer to the underlying data.
     * @return Const pointer to data (CPU or GPU based on current location)
     */
#ifdef CUDA_AVAILABLE
    const float* get_data() const {
        return is_on_gpu_ ? gpu_data_ : data_.data();
    }
    float* get_data() {
        return is_on_gpu_ ? gpu_data_ : data_.data();
    }
#else
    const float* get_data() const {
        return data_.data();
    }
    float* get_data() {
        return data_.data();
    }
#endif

    /**
     * @brief Resizes the matrix.
     * @param new_rows New number of rows
     * @param new_cols New number of columns
     */
    void resize(size_t new_rows, size_t new_cols);

    /**
     * @brief Element access operator.
     * @param row Row index
     * @param col Column index
     * @return Reference to the element
     */
    float& operator()(size_t row, size_t col);
    const float& operator()(size_t row, size_t col) const;

    /**
     * @brief Safe element access with bounds checking.
     * @param row Row index
     * @param col Column index
     * @return Reference to the element
     * @throws std::out_of_range if indices are invalid
     */
    float& at(size_t row, size_t col);
    const float& at(size_t row, size_t col) const;

    /**
     * @brief Gets a row as a vector.
     * @param row Row index
     * @return Vector containing the row
     */
    Vector row(size_t row) const;

    /**
     * @brief Sets a row from a vector.
     * @param row Row index
     * @param vec Vector containing new values
     */
    void set_row(size_t row, const Vector& vec);

    /**
     * @brief Computes the matrix transpose.
     * @return Transposed matrix
     */
    Matrix transpose() const;

    /**
     * @brief Applies ReLU activation function element-wise.
     */
    void apply_relu();

    /**
     * @brief Applies GELU activation function element-wise.
     */
    void apply_gelu();

    /**
     * @brief Applies GELU derivative for backpropagation.
     * @param x Input matrix
     */
    void apply_gelu_derivative(const Matrix& x);

    /**
     * @brief Applies softmax function row-wise.
     */
    void apply_softmax();

    /**
     * @brief Adds a bias vector to each row.
     * @param bias Bias vector to add
     */
    void add_bias(const Vector& bias);

    /**
     * @brief Matrix addition assignment operator.
     * @param other Matrix to add
     * @return Reference to this matrix
     */
    Matrix& operator+=(const Matrix& other);

    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(float scalar);
    Matrix& operator/=(float scalar);
    Matrix& operator*=(const Matrix& other);
    void save(std::ostream& os) const;
    static Matrix load(std::istream& is);
    void randomize(float min_val, float max_val);
    Vector row_sum() const;
    void fill(float value);
    void fill(const Matrix& m, float value);

    // Add element-wise multiplication (Hadamard product)
    Matrix hadamard(const Matrix& other) const {
        if (rows() != other.rows() || cols() != other.cols()) {
            throw std::runtime_error("Matrix dimensions must match for hadamard product");
        }

        Matrix result(rows(), cols());
        for (size_t i = 0; i < size(); ++i) {
            result.data()[i] = data()[i] * other.data()[i];
        }
        return result;
    }

    // Returns a view into a block of the matrix
    Matrix block(size_t start_row, size_t start_col, size_t num_rows, size_t num_cols) const {
        Matrix result(num_rows, num_cols);
        for (size_t i = 0; i < num_rows; ++i) {
            for (size_t j = 0; j < num_cols; ++j) {
                result(i, j) = (*this)(start_row + i, start_col + j);
            }
        }
        return result;
    }

    // Only declare copy constructor and assignment operator
    Matrix(const Matrix& other) {
        data_ = other.data_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        shape_ = other.shape_;
        owns_data_ = other.owns_data_;
#ifdef CUDA_AVAILABLE
        if (other.is_on_gpu_) {
            cudaMalloc(&gpu_data_, data_.size() * sizeof(float));
            cudaMemcpy(gpu_data_, other.gpu_data_, data_.size() * sizeof(float),
                       cudaMemcpyDeviceToDevice);
            is_on_gpu_ = true;
        }
#endif
    }

    Matrix& operator=(const Matrix& other); // Declaration only

    // Move operations - full implementation
    Matrix(Matrix&& other) noexcept
        : data_(std::move(other.data_)), rows_(other.rows_), cols_(other.cols_),
          shape_(other.shape_), owns_data_(other.owns_data_) {
        // Zero out the source object
        other.rows_ = 0;
        other.cols_ = 0;
        other.shape_ = std::make_tuple(0, 0);
        other.owns_data_ = false;
    }

    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            rows_ = other.rows_;
            cols_ = other.cols_;
            shape_ = other.shape_;
            owns_data_ = other.owns_data_;

            other.rows_ = 0;
            other.cols_ = 0;
            other.shape_ = std::make_tuple(0, 0);
            other.owns_data_ = false;
        }
        return *this;
    }

#ifdef CUDA_AVAILABLE
    Matrix to_gpu() const {
        Matrix gpu_matrix = *this;
        if (!gpu_matrix.is_on_gpu_) {
            cudaError_t err;
            err = cudaMalloc(&gpu_matrix.gpu_data_, data_.size() * sizeof(float));
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA malloc failed: " +
                                         std::string(cudaGetErrorString(err)));
            }

            err = cudaMemcpy(gpu_matrix.gpu_data_, data_.data(), data_.size() * sizeof(float),
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                cudaFree(gpu_matrix.gpu_data_);
                throw std::runtime_error("CUDA memcpy H2D failed: " +
                                         std::string(cudaGetErrorString(err)));
            }
            gpu_matrix.is_on_gpu_ = true;
        }
        return gpu_matrix;
    }

    Matrix to_cpu() const {
        if (!is_on_gpu_)
            return *this;
        Matrix cpu_matrix = *this;
        cudaMemcpy(cpu_matrix.data_.data(), gpu_data_, data_.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
        return cpu_matrix;
    }

    const float* data() const {
        return is_on_gpu_ ? gpu_data_ : data_.data();
    }
    float* data() {
        return is_on_gpu_ ? gpu_data_ : data_.data();
    }
#else
    const float* data() const {
        return data_.data();
    }
    float* data() {
        return data_.data();
    }
#endif

    ~Matrix() {
#ifdef CUDA_AVAILABLE
        if (gpu_data_ != nullptr) {
            cudaFree(gpu_data_);
            gpu_data_ = nullptr;
        }
#endif
    }

    static Matrix from_vector(const std::vector<float>& vec) {
        Matrix mat(1, vec.size());
        std::copy(vec.begin(), vec.end(), mat.data());
        return mat;
    }

    // Forward method for compatibility with neural network layers
    Matrix& forward(const Matrix& input) {
        *this = input;  // Copy input to this matrix
        return *this;
    }

    /**
     * @brief Initialize matrix with random values using Xavier/Glorot initialization
     * @param scale Scaling factor for initialization
     * @throws std::runtime_error if matrix doesn't own its data
     */
    void initialize_random(float scale);

    /**
     * @brief Initialize matrix with a constant value
     * @param value Value to initialize all elements with
     * @throws std::runtime_error if matrix doesn't own its data
     */
    void initialize_constant(float value);
};

// Make to_vector inline to allow multiple definitions
inline std::vector<int> to_vector(const Matrix& m) {
    return std::vector<int>(m.data(), m.data() + m.size());
}

// Non-member operators
Matrix operator+(const Matrix& a, const Matrix& b);
Matrix operator-(const Matrix& a, const Matrix& b);
Matrix operator*(const Matrix& m, float scalar);
Matrix operator*(float scalar, const Matrix& m);
Matrix operator/(const Matrix& m, float scalar);
Matrix operator*(const Matrix& a, const Matrix& b);

// Matrix multiplication function
Matrix matmul(const Matrix& a, const Matrix& b);

inline std::ostream& operator<<(std::ostream& os, const std::tuple<size_t, size_t>& shape) {
    os << std::get<0>(shape) << "x" << std::get<1>(shape);
    return os;
}