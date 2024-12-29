#include "../include/components.hpp"
#include <algorithm>
#include <cmath>

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

Matrix operator+(const Matrix &a, const Matrix &b) {
  Matrix result = a;
  result += b;
  return result;
}

Matrix matmul(const Matrix &a, const Matrix &b) {
  if (a.cols() != b.rows()) {
    throw std::invalid_argument(
        "Matrix dimensions don't match for multiplication");
  }

  Matrix result(a.rows(), b.cols(), 0.0f);
  for (size_t i = 0; i < a.rows(); ++i) {
    for (size_t j = 0; j < b.cols(); ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < a.cols(); ++k) {
        sum += a(i, k) * b(k, j);
      }
      result(i, j) = sum;
    }
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