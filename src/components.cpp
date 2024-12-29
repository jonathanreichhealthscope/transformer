#include "../include/components.hpp"

// Constructor implementations
Matrix::Matrix(size_t rows, size_t cols, float default_value)
    : data_(rows * cols, default_value), rows_(rows), cols_(cols) {}

Matrix::Matrix(const std::initializer_list<std::initializer_list<float>> &list)
    : rows_(list.size()), cols_(list.begin()->size()), data_(rows_ * cols_) {
  size_t i = 0;
  for (const auto &row : list) {
    if (row.size() != cols_) {
      throw std::invalid_argument("All rows must have the same size");
    }
    std::copy(row.begin(), row.end(), data_.begin() + i * cols_);
    i++;
  }
}

// Element access implementations
float &Matrix::operator()(size_t row, size_t col) {
  return data_[index(row, col)];
}

const float &Matrix::operator()(size_t row, size_t col) const {
  return data_[index(row, col)];
}

float &Matrix::at(size_t row, size_t col) {
  if (row >= rows_ || col >= cols_) {
    throw std::out_of_range("Matrix indices out of range");
  }
  return (*this)(row, col);
}

const float &Matrix::at(size_t row, size_t col) const {
  if (row >= rows_ || col >= cols_) {
    throw std::out_of_range("Matrix indices out of range");
  }
  return (*this)(row, col);
}
