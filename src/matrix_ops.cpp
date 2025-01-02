#include "../include/components.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

Vector Matrix::row(size_t row_idx) const {
  Vector result(cols_);
  for (size_t i = 0; i < cols_; ++i) {
    result[i] = (*this)(row_idx, i);
  }
  return result;
}

Matrix &Matrix::operator+=(const Matrix &other) {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::invalid_argument("Matrix dimensions don't match for addition");
  }
  for (size_t i = 0; i < data_.size(); ++i) {
    data_[i] += other.data_[i];
  }
  return *this;
}

Matrix &Matrix::operator-=(const Matrix &other) {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::runtime_error("Matrix dimensions don't match for subtraction");
  }

  for (size_t i = 0; i < data_.size(); ++i) {
    data_[i] -= other.data_[i];
  }
  return *this;
}

void Matrix::apply_softmax() {
  for (size_t i = 0; i < rows_; ++i) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < cols_; ++j) {
      max_val = std::max(max_val, (*this)(i, j));
    }

    float sum = 0.0f;
    for (size_t j = 0; j < cols_; ++j) {
      float &val = (*this)(i, j);
      val = std::exp(val - max_val);
      sum += val;
    }

    for (size_t j = 0; j < cols_; ++j) {
      (*this)(i, j) /= sum;
    }
  }
}

Matrix Matrix::transpose() const {
  Matrix result(cols_, rows_);
  for (size_t i = 0; i < rows_; ++i) {
    for (size_t j = 0; j < cols_; ++j) {
      result(j, i) = (*this)(i, j);
    }
  }
  return result;
}

Matrix operator*(const Matrix &m, float scalar) {
  Matrix result(m.rows(), m.cols());
  for (size_t i = 0; i < m.rows(); ++i) {
    for (size_t j = 0; j < m.cols(); ++j) {
      result(i, j) = m(i, j) * scalar;
    }
  }
  return result;
}

Matrix operator*(float scalar, const Matrix &m) { return m * scalar; }

Matrix operator+(const Matrix &a, const Matrix &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    throw std::runtime_error("Matrix dimensions don't match for addition");
  }
  Matrix result(a.rows(), a.cols());
  for (size_t i = 0; i < a.rows(); ++i) {
    for (size_t j = 0; j < a.cols(); ++j) {
      result(i, j) = a(i, j) + b(i, j);
    }
  }
  return result;
}

Matrix operator-(const Matrix &a, const Matrix &b) { return a + (b * -1.0f); }

Matrix operator/(const Matrix &m, float scalar) { return m * (1.0f / scalar); }

Matrix matmul(const Matrix &a, const Matrix &b) {
  if (a.cols() != b.rows()) {
    throw std::runtime_error(
        "Matrix dimensions don't match for multiplication");
  }
  float max_val = 0.0f;
  Matrix result(a.rows(), b.cols(), 0.0f);
  for (size_t i = 0; i < a.rows(); ++i) {
    for (size_t j = 0; j < b.cols(); ++j) {
      for (size_t k = 0; k < a.cols(); ++k) {
        result(i, j) += a(i, k) * b(k, j);
        max_val = std::max(max_val, std::abs(result(i, j)));
      }
    }
  }
  // Validate result is non-zero
  if(max_val == 0.0f) {
    std::cerr << "Warning: Matrix multiplication resulted in all zeros\n";
    std::cerr << "Input matrix A stats: min=" << a.min() << " max=" << a.max() << "\n";
    std::cerr << "Input matrix B stats: min=" << b.min() << " max=" << b.max() << "\n";
  }

  return result;
}

Matrix &Matrix::operator*=(float scalar) {
  for (float &val : data_) {
    val *= scalar;
  }
  return *this;
}

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