#include "../include/matrix.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <string>
// Constructor implementations
Matrix::Matrix() : rows_(0), cols_(0) {}

Matrix::Matrix(size_t rows, size_t cols, float init_val)
    : data_(rows * cols, init_val), rows_(rows), cols_(cols) {}

Matrix::Matrix(size_t rows, size_t cols, float *external_data)
    : rows_(rows), cols_(cols) {
  data_.assign(external_data, external_data + (rows * cols));
}

Matrix::Matrix(size_t rows, size_t cols, float* external_data, bool is_owner) 
    : rows_(rows), cols_(cols), shape_(std::make_tuple(rows, cols)), owns_data_(is_owner) {
    if (is_owner) {
        // If we own the data, copy it to our vector
        data_.assign(external_data, external_data + (rows * cols));
    } else {
        // If we don't own the data, just point to it
        data_ = std::vector<float>(external_data, external_data + (rows * cols));
    }
} 

// Basic operations
void Matrix::resize(size_t new_rows, size_t new_cols) {
  if (new_rows == rows_ && new_cols == cols_) {
    return;
  }
  data_.resize(new_rows * new_cols);
  rows_ = new_rows;
  cols_ = new_cols;
}

float &Matrix::operator()(size_t row, size_t col) {
  if (row >= rows_ || col >= cols_) {
    throw std::out_of_range("Matrix index out of bounds");
  }
  return data_[row * cols_ + col];
}

const float &Matrix::operator()(size_t row, size_t col) const {
  if (row >= rows_ || col >= cols_) {
    throw std::out_of_range("Matrix index out of bounds");
  }
  return data_[row * cols_ + col];
}

float &Matrix::at(size_t row, size_t col) { return operator()(row, col); }

const float &Matrix::at(size_t row, size_t col) const {
  return operator()(row, col);
}

// Row operations
Vector Matrix::row(size_t row) const {
  if (row >= rows_) {
    throw std::out_of_range("Row index out of bounds");
  }
  return Vector(data_.begin() + row * cols_, data_.begin() + (row + 1) * cols_);
}

void Matrix::set_row(size_t row, const Vector &vec) {
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
  for (size_t i = 0; i < rows_; ++i) {
    for (size_t j = 0; j < cols_; ++j) {
      result(j, i) = (*this)(i, j);
    }
  }
  return result;
}

void Matrix::apply_relu() {
  for (float &val : data_) {
    val = std::max(0.0f, val);
  }
}

void Matrix::apply_gelu() {
  constexpr float sqrt_2_over_pi = 0.7978845608028654f;
  for (float &val : data_) {
    float cdf = 0.5f * (1.0f + std::tanh(sqrt_2_over_pi *
                                         (val + 0.044715f * val * val * val)));
    val *= cdf;
  }
}

void Matrix::apply_softmax() {
  for (size_t i = 0; i < rows_; ++i) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < cols_; ++j) {
      max_val = std::max(max_val, (*this)(i, j));
    }

    float sum = 0.0f;
    for (size_t j = 0; j < cols_; ++j) {
      float exp_val = std::exp((*this)(i, j) - max_val);
      (*this)(i, j) = exp_val;
      sum += exp_val;
    }

    for (size_t j = 0; j < cols_; ++j) {
      (*this)(i, j) /= sum;
    }
  }
}

void Matrix::add_bias(const Vector &bias) {
  if (bias.size() != cols_) {
    throw std::invalid_argument("Bias size must match matrix columns");
  }
  for (size_t i = 0; i < rows_; ++i) {
    for (size_t j = 0; j < cols_; ++j) {
      (*this)(i, j) += bias[j];
    }
  }
}

// Operator implementations
Matrix &Matrix::operator+=(const Matrix &other) {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::invalid_argument("Matrix dimensions must match for addition");
  }
  for (size_t i = 0; i < data_.size(); ++i) {
    data_[i] += other.data_[i];
  }
  return *this;
}

Matrix &Matrix::operator-=(const Matrix &other) {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::invalid_argument("Matrix dimensions must match for subtraction");
  }
  for (size_t i = 0; i < data_.size(); ++i) {
    data_[i] -= other.data_[i];
  }
  return *this;
}

Matrix &Matrix::operator*=(float scalar) {
  for (float &val : data_) {
    val *= scalar;
  }
  return *this;
}

Matrix &Matrix::operator/=(float scalar) {
  if (scalar == 0.0f) {
    throw std::invalid_argument("Division by zero");
  }
  for (float &val : data_) {
    val /= scalar;
  }
  return *this;
}

Matrix &Matrix::operator*=(const Matrix &other) {
  if (cols_ != other.rows_) {
    throw std::invalid_argument("Invalid matrix dimensions for multiplication");
  }
  Matrix result(rows_, other.cols_);
  for (size_t i = 0; i < rows_; ++i) {
    for (size_t j = 0; j < other.cols_; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < cols_; ++k) {
        sum += (*this)(i, k) * other(k, j);
      }
      result(i, j) = sum;
    }
  }
  *this = std::move(result);
  return *this;
}

// Serialization
void Matrix::save(std::ostream &os) const {
  os.write(reinterpret_cast<const char *>(&rows_), sizeof(rows_));
  os.write(reinterpret_cast<const char *>(&cols_), sizeof(cols_));
  os.write(reinterpret_cast<const char *>(data_.data()),
           data_.size() * sizeof(float));
}

Matrix Matrix::load(std::istream &is) {
  size_t rows, cols;
  is.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  is.read(reinterpret_cast<char *>(&cols), sizeof(cols));
  Matrix result(rows, cols);
  is.read(reinterpret_cast<char *>(result.data_.data()),
          result.data_.size() * sizeof(float));
  return result;
}

// Utility functions
void Matrix::randomize(float min_val, float max_val) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min_val, max_val);
  for (float &val : data_) {
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
Matrix operator+(const Matrix &a, const Matrix &b) {
  Matrix result = a;
  result += b;
  return result;
}

Matrix operator-(const Matrix &a, const Matrix &b) {
  Matrix result = a;
  result -= b;
  return result;
}

Matrix operator*(const Matrix &m, float scalar) {
  Matrix result = m;
  result *= scalar;
  return result;
}

Matrix operator*(float scalar, const Matrix &m) { return m * scalar; }

Matrix operator/(const Matrix &m, float scalar) {
  Matrix result = m;
  result /= scalar;
  return result;
}

Matrix operator*(const Matrix &a, const Matrix &b) {
  Matrix result = a;
  result *= b;
  return result;
}

Matrix matmul(const Matrix &a, const Matrix &b) {
  /*std::cout << "Matrix multiplication dimensions:" << std::endl;
  std::cout << "A: " << a.rows() << "x" << a.cols() << std::endl;
  std::cout << "B: " << b.rows() << "x" << b.cols() << std::endl;*/

  if (a.cols() != b.rows()) {
    throw std::runtime_error("Invalid matrix dimensions for multiplication: " +
                             std::to_string(a.cols()) +
                             " != " + std::to_string(b.rows()));
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
