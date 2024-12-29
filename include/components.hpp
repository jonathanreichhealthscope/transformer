#pragma once
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

class Vector {
private:
  std::vector<float> data_;
  size_t size_;

public:
  // Constructors
  Vector() : size_(0) {}
  explicit Vector(size_t size, float default_value = 0.0f);
  Vector(const std::initializer_list<float> &list);

  // Element access
  float &operator[](size_t index) { return data_[index]; }
  const float &operator[](size_t index) const { return data_[index]; }
  float &at(size_t index);
  const float &at(size_t index) const;

  // Capacity
  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  // Operations
  Vector &operator+=(const Vector &other);
  Vector &operator-=(const Vector &other);
  Vector &operator*=(float scalar);
  Vector &operator/=(float scalar);

  // Mathematical operations
  float dot(const Vector &other) const;
  float norm() const;
  void normalize();

  // Iterator support
  auto begin() { return data_.begin(); }
  auto end() { return data_.end(); }
  auto begin() const { return data_.begin(); }
  auto end() const { return data_.end(); }

  // Concatenation
  static Vector concatenate(const Vector &a, const Vector &b) {
    Vector result(a.size() + b.size());
    std::copy(a.begin(), a.end(), result.begin());
    std::copy(b.begin(), b.end(), result.begin() + a.size());
    return result;
  }

  // Add data accessors
  float *data() { return data_.data(); }
  const float *data() const { return data_.data(); }

  void save(std::ostream &os) const;
  static Vector load(std::istream &is);

  void randomize(float min_val, float max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (auto &val : data_) {
      val = dis(gen);
    }
  }
};

// Vector operations
Vector operator+(const Vector &a, const Vector &b);
Vector operator-(const Vector &a, const Vector &b);
Vector operator*(const Vector &v, float scalar);
Vector operator*(float scalar, const Vector &v);
Vector operator/(const Vector &v, float scalar);

class Matrix {
private:
  std::vector<float> data_;
  size_t rows_;
  size_t cols_;

  // Helper to get index in flat array
  size_t index(size_t row, size_t col) const { return row * cols_ + col; }

public:
  // Constructors
  Matrix() : rows_(0), cols_(0) {}
  Matrix(size_t rows, size_t cols, float default_value = 0.0f);
  Matrix(const std::initializer_list<std::initializer_list<float>> &list);

  // Element access
  float &operator()(size_t row, size_t col);
  const float &operator()(size_t row, size_t col) const;
  float &at(size_t row, size_t col);
  const float &at(size_t row, size_t col) const;

  // Row access
  Vector row(size_t row) const;
  void set_row(size_t row, const Vector &vec);

  // Capacity
  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  bool empty() const { return rows_ == 0 || cols_ == 0; }

  // Data access
  float *data() { return data_.data(); }
  const float *data() const { return data_.data(); }

  // Mathematical operations
  Matrix transpose() const;
  void apply_relu();
  void apply_gelu();
  void apply_softmax();
  void add_bias(const Vector &bias);
  static Matrix concatenate(const Matrix &a, const Matrix &b);
  static Matrix matmul(const Matrix &a, const Matrix &b);

  // Operators
  Matrix &operator+=(const Matrix &other);
  Matrix &operator-=(const Matrix &other);
  Matrix &operator*=(float scalar);
  Matrix &operator/=(float scalar);
  Matrix &operator*=(const Matrix &other); // Hadamard product

  // Serialization
  void save(std::ostream &os) const;
  static Matrix load(std::istream &is);

  void randomize(float min_val, float max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (auto &val : data_) {
      val = dis(gen);
    }
  }

  Vector row_sum() const {
    Vector result(rows_);
    for (size_t i = 0; i < rows_; ++i) {
      float sum = 0.0f;
      for (size_t j = 0; j < cols_; ++j) {
        sum += (*this)(i, j);
      }
      result[i] = sum;
    }
    return result;
  }
};

// Non-member operators
Matrix operator+(const Matrix &a, const Matrix &b);
Matrix operator-(const Matrix &a, const Matrix &b);
Matrix operator*(const Matrix &m, float scalar);
Matrix operator*(float scalar, const Matrix &m);
Matrix operator/(const Matrix &m, float scalar);
Matrix operator*(const Matrix &a, const Matrix &b); // Hadamard product
Matrix matmul(const Matrix &a, const Matrix &b);