#include "components.hpp"
#include <algorithm>
#define M_PI 3.14159265358979323846

// Vector Implementation
Vector::Vector(size_t size, float default_value) 
    : data_(size, default_value), size_(size) {}

Vector::Vector(const std::initializer_list<float>& list) 
    : data_(list), size_(list.size()) {}

float& Vector::at(size_t index) {
    return data_[index];
}

const float& Vector::at(size_t index) const {
    return data_[index];
}

Vector& Vector::operator+=(const Vector& other) {
    if (size_ != other.size_) {
        throw std::invalid_argument("Vector sizes don't match");
    }
    for (size_t i = 0; i < size_; ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Vector& Vector::operator-=(const Vector& other) {
    if (size_ != other.size_) {
        throw std::invalid_argument("Vector sizes don't match");
    }
    for (size_t i = 0; i < size_; ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

Vector& Vector::operator*=(float scalar) {
    for (float& val : data_) {
        val *= scalar;
    }
    return *this;
}

Vector& Vector::operator/=(float scalar) {
    if (scalar == 0.0f) {
        throw std::invalid_argument("Division by zero");
    }
    for (float& val : data_) {
        val /= scalar;
    }
    return *this;
}

float Vector::dot(const Vector& other) const {
    if (size_ != other.size_) {
        throw std::invalid_argument("Vector sizes don't match");
    }
    float result = 0.0f;
    for (size_t i = 0; i < size_; ++i) {
        result += data_[i] * other.data_[i];
    }
    return result;
}

float Vector::norm() const {
    return std::sqrt(dot(*this));
}

void Vector::normalize() {
    float n = norm();
    if (n > 0.0f) {
        *this /= n;
    }
}

// Vector operations
Vector operator+(const Vector& a, const Vector& b) {
    Vector result = a;
    result += b;
    return result;
}

Vector operator-(const Vector& a, const Vector& b) {
    Vector result = a;
    result -= b;
    return result;
}

Vector operator*(const Vector& v, float scalar) {
    Vector result = v;
    result *= scalar;
    return result;
}

Vector operator*(float scalar, const Vector& v) {
    return v * scalar;
}

Vector operator/(const Vector& v, float scalar) {
    Vector result = v;
    result /= scalar;
    return result;
}

// Matrix Implementation
Matrix::Matrix(size_t rows, size_t cols, float default_value) 
    : data_(rows, Vector(cols, default_value)), rows_(rows), cols_(cols) {}

Matrix::Matrix(const std::initializer_list<std::initializer_list<float>>& list) 
    : rows_(list.size()), cols_(list.begin()->size()) {
    data_.reserve(rows_);
    for (const auto& row : list) {
        data_.emplace_back(row);
    }
}

Matrix Matrix::concatenate(const Matrix& a, const Matrix& b) {
    if (a.rows_ != b.rows_) throw std::invalid_argument("Matrix rows don't match");
    Matrix result(a.rows_, a.cols_ + b.cols_);
    for (size_t i = 0; i < a.rows_; ++i) {
        result[i] = Vector::concatenate(a[i], b[i]);
    }
    return result;
}

float& Matrix::at(size_t row, size_t col) {
    return data_[row][col];
}

const float& Matrix::at(size_t row, size_t col) const {
    return data_[row][col];
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions don't match");
    }
    for (size_t i = 0; i < rows_; ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result[j][i] = data_[i][j];
        }
    }
    return result;
}

Matrix Matrix::matmul(const Matrix& a, const Matrix& b) {
    if (a.cols_ != b.rows_) 
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    
    Matrix result(a.rows_, b.cols_, 0.0f);
    for (size_t i = 0; i < a.rows_; ++i) {
        for (size_t j = 0; j < b.cols_; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < a.cols_; ++k) {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

void Matrix::add_bias(const Vector& bias) {
    if (bias.size() != cols_) {
        throw std::invalid_argument("Bias dimension doesn't match");
    }
    for (auto& row : data_) {
        row += bias;
    }
}

void Matrix::apply_relu() {
    for (auto& row : data_) {
        for (float& val : row) {
            val = std::max(0.0f, val);
        }
    }
}

void Matrix::apply_gelu() {
    const float sqrt2_over_pi = std::sqrt(2.0f / M_PI);
    for (auto& row : data_) {
        for (float& val : row) {
            val = 0.5f * val * (1.0f + std::tanh(sqrt2_over_pi * (val + 0.044715f * std::pow(val, 3))));
        }
    }
}

void Matrix::apply_softmax() {
    for (auto& row : data_) {
        float max_val = *std::max_element(row.begin(), row.end());
        float sum = 0.0f;
        
        for (float& val : row) {
            val = std::exp(val - max_val);
            sum += val;
        }
        
        for (float& val : row) {
            val /= sum;
        }
    }
}

// Element-wise multiplication (Hadamard product)
Matrix operator*(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        throw std::invalid_argument("Matrix dimensions don't match for Hadamard product");
    }
    
    Matrix result(a.rows(), a.cols());
    for (size_t i = 0; i < a.rows(); ++i) {
        for (size_t j = 0; j < a.cols(); ++j) {
            result(i, j) = a(i, j) * b(i, j);
        }
    }
    return result;
}

// Element-wise multiplication assignment
Matrix& Matrix::operator*=(const Matrix& other) {
    if (rows_ != other.rows() || cols_ != other.cols()) {
        throw std::invalid_argument("Matrix dimensions don't match for element-wise multiplication");
    }
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            data_[i][j] *= other(i, j);
        }
    }
    return *this;
}

void Matrix::save(std::ostream& os) const {
    // Save dimensions
    os.write(reinterpret_cast<const char*>(&rows_), sizeof(rows_));
    os.write(reinterpret_cast<const char*>(&cols_), sizeof(cols_));
    
    // Save data
    for (size_t i = 0; i < rows_; ++i) {
        os.write(reinterpret_cast<const char*>(data_[i].data()), 
                cols_ * sizeof(float));
    }
}

Matrix Matrix::load(std::istream& is) {
    size_t rows, cols;
    is.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    is.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        is.read(reinterpret_cast<char*>(result.data_[i].data()), 
               cols * sizeof(float));
    }
    
    return result;
} 