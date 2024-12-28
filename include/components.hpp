#pragma once
#include <vector>
#include <stdexcept>
#include <memory>
#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>

class Vector {
private:
    std::vector<float> data_;
    size_t size_;

public:
    // Constructors
    Vector() : size_(0) {}
    explicit Vector(size_t size, float default_value = 0.0f);
    Vector(const std::initializer_list<float>& list);
    
    // Element access
    float& operator[](size_t index) { return data_[index]; }
    const float& operator[](size_t index) const { return data_[index]; }
    float& at(size_t index);
    const float& at(size_t index) const;
    
    // Capacity
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    
    // Operations
    Vector& operator+=(const Vector& other);
    Vector& operator-=(const Vector& other);
    Vector& operator*=(float scalar);
    Vector& operator/=(float scalar);
    
    // Mathematical operations
    float dot(const Vector& other) const;
    float norm() const;
    void normalize();
    
    // Iterator support
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }
    
    // Concatenation
    static Vector concatenate(const Vector& a, const Vector& b) {
        Vector result(a.size() + b.size());
        std::copy(a.begin(), a.end(), result.begin());
        std::copy(b.begin(), b.end(), result.begin() + a.size());
        return result;
    }
    
    // Add data accessors
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
};

// Vector operations
Vector operator+(const Vector& a, const Vector& b);
Vector operator-(const Vector& a, const Vector& b);
Vector operator*(const Vector& v, float scalar);
Vector operator*(float scalar, const Vector& v);
Vector operator/(const Vector& v, float scalar);

class Matrix {
private:
    std::vector<Vector> data_;
    size_t rows_;
    size_t cols_;

public:
    // Constructors
    Matrix() : rows_(0), cols_(0) {}
    Matrix(size_t rows, size_t cols, float default_value = 0.0f);
    Matrix(const std::initializer_list<std::initializer_list<float>>& list);
    
    // Element access
    Vector& operator[](size_t index) { return data_[index]; }
    const Vector& operator[](size_t index) const { return data_[index]; }
    float& at(size_t row, size_t col);
    const float& at(size_t row, size_t col) const;
    
    // Capacity
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    bool empty() const { return rows_ == 0 || cols_ == 0; }
    
    // Operations
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(float scalar);
    Matrix& operator/=(float scalar);
    
    // Mathematical operations
    Matrix transpose() const;
    static Matrix matmul(const Matrix& a, const Matrix& b);
    void add_bias(const Vector& bias);
    
    // Neural network specific operations
    void apply_relu();
    void apply_gelu();
    void apply_softmax();
    
    // Iterator support
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }
    
    // Block operations
    Matrix block(size_t start_row, size_t end_row) const {
        if (start_row >= rows_ || end_row > rows_ || start_row >= end_row) {
            throw std::out_of_range("Invalid block range");
        }
        
        Matrix result(end_row - start_row, cols_);
        for (size_t i = 0; i < end_row - start_row; ++i) {
            result[i] = data_[start_row + i];
        }
        return result;
    }
    
    // Matrix row operations
    Matrix row(size_t index) const {
        if (index >= rows_) {
            throw std::out_of_range("Row index out of range");
        }
        Matrix result(1, cols_);
        result[0] = data_[index];
        return result;
    }
    
    void set_row(size_t index, const Matrix& row) {
        if (index >= rows_ || row.rows() != 1 || row.cols() != cols_) {
            throw std::out_of_range("Invalid row dimensions");
        }
        data_[index] = row[0];
    }
    
    static Matrix concatenate(const Matrix& a, const Matrix& b);
    
    // Add these operator() overloads
    float& operator()(size_t row, size_t col) {
        return data_[row][col];
    }
    
    const float& operator()(size_t row, size_t col) const {
        return data_[row][col];
    }
    
    // Add these data accessors
    float* data() { return &data_[0][0]; }
    const float* data() const { return &data_[0][0]; }
    
    // Add operator*= for matrix multiplication
    Matrix& operator*=(const Matrix& other);
    
    // Add serialization methods
    void save(std::ostream& os) const;
    static Matrix load(std::istream& is);
};

// Matrix operations
Matrix operator+(const Matrix& a, const Matrix& b);
Matrix operator-(const Matrix& a, const Matrix& b);
Matrix operator*(const Matrix& m, float scalar);
Matrix operator*(float scalar, const Matrix& m);
Matrix operator/(const Matrix& m, float scalar);
Matrix operator*(const Matrix& a, const Matrix& b);  // Hadamard product