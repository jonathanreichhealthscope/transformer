#include "../include/matrix.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <string>
// Constructor implementations
Matrix::Matrix() : rows_(0), cols_(0), shape_(std::make_tuple(0, 0)) {}

Matrix::Matrix(size_t rows, size_t cols, float init_val) {
    // Check for zero dimensions
    if (rows == 0 || cols == 0) {
        throw std::runtime_error("Matrix dimensions cannot be zero");
    }

    // Check for overflow in size calculation
    if (rows > SIZE_MAX / cols) {
        throw std::runtime_error("Matrix dimensions too large - would cause overflow");
    }

    // Check total size is reasonable
    size_t total_size = rows * cols;
    if (total_size > 1000000000) { // 1 billion elements max
        throw std::runtime_error("Matrix dimensions too large - exceeds maximum allowed size");
    }

    try {
        data_.resize(total_size, init_val);
    } catch (const std::bad_alloc& e) {
        throw std::runtime_error("Failed to allocate memory for matrix: " + std::string(e.what()));
    } catch (const std::length_error& e) {
        throw std::runtime_error("Matrix dimensions too large: " + std::string(e.what()));
    }

    rows_ = rows;
    cols_ = cols;
    shape_ = std::make_tuple(rows, cols);
    owns_data_ = true;
}

Matrix::Matrix(size_t rows, size_t cols, float* external_data)
    : data_(external_data, external_data + rows * cols), rows_(rows), cols_(cols),
      shape_(std::make_tuple(rows, cols)), owns_data_(false) {}

Matrix::Matrix(size_t rows, size_t cols, float* external_data, bool is_owner)
    : data_(external_data, external_data + rows * cols), rows_(rows), cols_(cols),
      shape_(std::make_tuple(rows, cols)), owns_data_(is_owner) {}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        if (other.empty()) {
            data_.clear();
            rows_ = 0;
            cols_ = 0;
            shape_ = std::make_tuple(0, 0);
            owns_data_ = true;
            return *this;
        }

        try {
            data_.resize(other.data_.size());
            std::copy(other.data_.begin(), other.data_.end(), data_.begin());
            rows_ = other.rows_;
            cols_ = other.cols_;
            shape_ = other.shape_;
            owns_data_ = true;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to assign matrix: " + std::string(e.what()));
        }
    }
    return *this;
}

// Basic operations
void Matrix::resize(size_t new_rows, size_t new_cols) {
    // Check for no-op resize
    if (new_rows == rows_ && new_cols == cols_) {
        return;
    }

    // Check for overflow
    if (new_rows > SIZE_MAX / new_cols) {
        throw std::runtime_error("Matrix dimensions would cause overflow");
    }

    size_t new_size = new_rows * new_cols;

    try {
        // Create new vector with new size
        std::vector<float> new_data(new_size, 0.0f);

        // Copy existing data if possible
        size_t min_rows = std::min(rows_, new_rows);
        size_t min_cols = std::min(cols_, new_cols);

        for (size_t i = 0; i < min_rows; ++i) {
            for (size_t j = 0; j < min_cols; ++j) {
                new_data[i * new_cols + j] = data_[i * cols_ + j];
            }
        }

        // Swap the new data into place
        data_.swap(new_data);
        rows_ = new_rows;
        cols_ = new_cols;
        shape_ = std::make_tuple(new_rows, new_cols);

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to resize matrix: " + std::string(e.what()));
    }
}

float& Matrix::operator()(size_t row, size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data_[row * cols_ + col];
}

const float& Matrix::operator()(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data_[row * cols_ + col];
}

float& Matrix::at(size_t row, size_t col) {
    return operator()(row, col);
}

const float& Matrix::at(size_t row, size_t col) const {
    return operator()(row, col);
}

// Row operations
Vector Matrix::row(size_t row) const {
    if (row >= rows_) {
        throw std::out_of_range("Row index out of bounds");
    }
    return Vector(data_.begin() + row * cols_, data_.begin() + (row + 1) * cols_);
}

void Matrix::set_row(size_t row, const Vector& vec) {
    if (row >= rows_) {
        throw std::out_of_range("Row index out of bounds");
    }
    if (vec.size() != cols_) {
        throw std::invalid_argument("Vector size must match matrix columns");
    }
    std::copy(vec.begin(), vec.end(), data_.begin() + row * cols_);
}

// Matrix operations
Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

void Matrix::apply_relu() {
#pragma omp parallel for simd
    for (size_t i = 0; i < data_.size(); i++) {
        data_[i] = std::max(0.0f, data_[i]);
    }
}

void Matrix::apply_gelu() {
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
#pragma omp parallel for simd
    for (size_t i = 0; i < data_.size(); i++) {
        float val = data_[i];
        float cdf = 0.5f * (1.0f + std::tanh(sqrt_2_over_pi * (val + 0.044715f * val * val * val)));
        data_[i] = val * cdf;
    }
}

void Matrix::apply_gelu_derivative(const Matrix& x) {
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    if (size() != x.size()) {
        throw std::runtime_error("Matrix dimensions must match for GELU derivative");
    }

    if (data_.empty() || x.data_.empty()) {
        throw std::runtime_error("Empty matrix in GELU derivative");
    }

#pragma omp parallel for simd
    for (size_t i = 0; i < size(); i++) {
        if (i >= x.data_.size() || i >= data_.size()) {
            throw std::runtime_error("Index out of bounds in GELU derivative");
        }

        float val = x.data_[i];
        val = std::clamp(val, -10.0f, 10.0f);

        float cdf = 0.5f * (1.0f + std::tanh(sqrt_2_over_pi * (val + 0.044715f * val * val * val)));
        float pdf = sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * val * val);
        float derivative = cdf + val * pdf * (1.0f - std::tanh(val) * std::tanh(val));
        derivative = std::clamp(derivative, -10.0f, 10.0f);
        data_[i] *= derivative;
    }
}

void Matrix::apply_softmax() {
#pragma omp parallel for
    for (size_t i = 0; i < rows_; ++i) {
        float max_val = -std::numeric_limits<float>::infinity();
#pragma omp simd reduction(max : max_val)
        for (size_t j = 0; j < cols_; ++j) {
            max_val = std::max(max_val, (*this)(i, j));
        }

        float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
        for (size_t j = 0; j < cols_; ++j) {
            float exp_val = std::exp((*this)(i, j) - max_val);
            (*this)(i, j) = exp_val;
            sum += exp_val;
        }

#pragma omp simd
        for (size_t j = 0; j < cols_; ++j) {
            (*this)(i, j) /= sum;
        }
    }
}

void Matrix::add_bias(const Vector& bias) {
    if (bias.size() != cols_) {
        throw std::invalid_argument("Bias size must match matrix columns");
    }
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            (*this)(i, j) += bias[j];
        }
    }
}

void Matrix::fill(float value) {
    if (data_.empty()) {
        throw std::runtime_error("Cannot fill empty matrix");
    }
    std::fill(data_.begin(), data_.end(), value);
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
#pragma omp parallel for simd
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Matrix& Matrix::operator*=(float scalar) {
#pragma omp parallel for simd
    for (size_t i = 0; i < data_.size(); i++) {
        data_[i] *= scalar;
    }
    return *this;
}

Matrix& Matrix::operator/=(float scalar) {
    if (scalar == 0.0f) {
        throw std::invalid_argument("Division by zero");
    }
#pragma omp parallel for simd
    for (size_t i = 0; i < data_.size(); i++) {
        data_[i] /= scalar;
    }
    return *this;
}

Matrix& Matrix::operator*=(const Matrix& other) {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Invalid matrix dimensions for multiplication");
    }
    Matrix result(rows_, other.cols_);

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < other.cols_; ++j) {
            float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
            for (size_t k = 0; k < cols_; ++k) {
                sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) = sum;
        }
    }
    *this = std::move(result);
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
#pragma omp parallel for simd
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

// Serialization
void Matrix::save(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&rows_), sizeof(rows_));
    os.write(reinterpret_cast<const char*>(&cols_), sizeof(cols_));
    os.write(reinterpret_cast<const char*>(data_.data()), data_.size() * sizeof(float));
}

Matrix Matrix::load(std::istream& is) {
    size_t rows, cols;
    is.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    is.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    Matrix result(rows, cols);
    is.read(reinterpret_cast<char*>(result.data_.data()), result.data_.size() * sizeof(float));
    return result;
}

// Utility functions
void Matrix::randomize(float min_val, float max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (float& val : data_) {
        val = dis(gen);
    }
}

Vector Matrix::row_sum() const {
    Vector result(cols_, 0.0f);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result[j] += (*this)(i, j);
        }
    }
    return result;
}

// Non-member operators
Matrix operator+(const Matrix& a, const Matrix& b) {
    Matrix result = a;
    result += b;
    return result;
}

Matrix operator-(const Matrix& a, const Matrix& b) {
    Matrix result = a;
    result -= b;
    return result;
}

Matrix operator*(const Matrix& m, float scalar) {
    Matrix result = m;
    result *= scalar;
    return result;
}

Matrix operator*(float scalar, const Matrix& m) {
    return m * scalar;
}

Matrix operator/(const Matrix& m, float scalar) {
    Matrix result = m;
    result /= scalar;
    return result;
}

Matrix operator*(const Matrix& a, const Matrix& b) {
    Matrix result = a;
    result *= b;
    return result;
}

Matrix matmul(const Matrix& a, const Matrix& b) {
    /*std::cout << "Matrix multiplication dimensions:" << std::endl;
    std::cout << "A: " << a.rows() << "x" << a.cols() << std::endl;
    std::cout << "B: " << b.rows() << "x" << b.cols() << std::endl;*/

    if (a.cols() != b.rows()) {
        throw std::runtime_error("Invalid matrix dimensions for multiplication: " +
                                 std::to_string(a.cols()) + " != " + std::to_string(b.rows()));
    }

    Matrix result(a.rows(), b.cols());

    for (size_t i = 0; i < a.rows(); i++) {
        for (size_t j = 0; j < b.cols(); j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < a.cols(); k++) {
                sum += a(i, k) * b(k, j);
            }
            result(i, j) = sum;
        }
    }

    return result;
}
