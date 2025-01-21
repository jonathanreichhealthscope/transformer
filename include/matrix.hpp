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
#define M_PI 3.14159265358979323846
// Forward declarations
class Matrix;
class Vector;

class Matrix {
  private:
    std::vector<float> data_;
    size_t rows_;
    size_t cols_;
    std::tuple<size_t, size_t> shape_;
    bool owns_data_ = true;
#ifdef CUDA_AVAILABLE
    float* gpu_data_ = nullptr;
    bool is_on_gpu_ = false;
#endif

  public:
    // Constructor declarations only
    Matrix();
    Matrix(size_t rows, size_t cols, float init_val = 0.0f);
    Matrix(size_t rows, size_t cols, float* external_data);
    Matrix(size_t rows, size_t cols, float* external_data, bool is_owner);
    Matrix(size_t rows, size_t cols, size_t batch_size, float* external_data);
    Matrix(size_t rows, size_t cols, const float* data);

    // Rest of the class interface
    size_t rows() const {
        return rows_;
    }
    size_t cols() const {
        return cols_;
    }
    size_t size() const {
        return data_.size();
    }
    size_t bytes() const {
        return size() * sizeof(float);
    }
    std::tuple<size_t, size_t> shape() const {
        return shape_;
    }
    bool empty() const {
        return data_.empty();
    }

    // Data access
    float min() const {
        return *std::min_element(data_.begin(), data_.end());
    }
    float max() const {
        return *std::max_element(data_.begin(), data_.end());
    }

// Single unified data access method
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

    // Matrix operations declarations
    void resize(size_t new_rows, size_t new_cols);
    float& operator()(size_t row, size_t col);
    const float& operator()(size_t row, size_t col) const;
    float& at(size_t row, size_t col);
    const float& at(size_t row, size_t col) const;
    Vector row(size_t row) const;
    void set_row(size_t row, const Vector& vec);
    Matrix transpose() const;
    void apply_relu();
    void apply_gelu();
    void apply_gelu_derivative(const Matrix& x);
    void apply_softmax();
    void add_bias(const Vector& bias);
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
};

// Make to_vector inline to allow multiple definitions
inline std::vector<int> to_vector(const Matrix& m) {
    return std::vector<int>(m.data(), m.data() + m.size());
}

class Vector {
  private:
    std::vector<float> data_;
    size_t size_;

  public:
    // Constructors (declarations only)
    Vector();
    Vector(size_t size, float default_value = 0.0f);
    Vector(const std::initializer_list<float>& list);
    template <typename Iterator>
    Vector(Iterator first, Iterator last) : data_(first, last), size_(std::distance(first, last)) {}

    // Data access
    float* data() {
        return data_.data();
    }
    const float* data() const {
        return data_.data();
    }
    size_t size() const {
        return size_;
    }

    // Element access
    float& operator[](size_t i) {
        return data_[i];
    }
    const float& operator[](size_t i) const {
        return data_[i];
    }

    // Modified operator+= to handle gradients properly
    Vector& operator+=(const Vector& other) {
        if (size_ != other.size()) {
            throw std::invalid_argument("Vector dimensions must match for addition");
        }
        for (size_t i = 0; i < size_; ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    // Iterator access
    auto begin() {
        return data_.begin();
    }
    auto end() {
        return data_.end();
    }
    auto begin() const {
        return data_.begin();
    }
    auto end() const {
        return data_.end();
    }

    // Utility functions
    bool empty() const {
        return data_.empty();
    }
    void resize(size_t new_size) {
        data_.resize(new_size);
        size_ = new_size;
    }
    void fill(float value) {
        std::fill(data_.begin(), data_.end(), value);
    }
    void randomize(float min_val, float max_val) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(min_val, max_val);
        for (float& val : data_) {
            val = dis(gen);
        }
    }

    // Serialization
    void save(std::ostream& os) const;
    static Vector load(std::istream& is);
};

// Non-member operators
Matrix operator+(const Matrix& a, const Matrix& b);
Matrix operator-(const Matrix& a, const Matrix& b);
Matrix operator*(const Matrix& m, float scalar);
Matrix operator*(float scalar, const Matrix& m);
Matrix operator/(const Matrix& m, float scalar);
Matrix operator*(const Matrix& a, const Matrix& b);
Matrix matmul(const Matrix& a, const Matrix& b);

inline std::ostream& operator<<(std::ostream& os, const std::tuple<size_t, size_t>& shape) {
    os << std::get<0>(shape) << "x" << std::get<1>(shape);
    return os;
}