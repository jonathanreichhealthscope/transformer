#pragma once
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <iostream>
#include <random>

// Forward declarations
class Matrix;
class Vector;

class Matrix {
private:
    std::vector<float> data_;
    size_t rows_;
    size_t cols_;

public:
    // Constructors
    Matrix();
    Matrix(size_t rows, size_t cols, float init_val = 0.0f);
    Matrix(size_t rows, size_t cols, float* external_data);

    // Size-related methods
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return data_.size(); }
    size_t bytes() const { return size() * sizeof(float); }

    // Matrix operations
    void resize(size_t new_rows, size_t new_cols);
    float& operator()(size_t row, size_t col);
    const float& operator()(size_t row, size_t col) const;
    float& at(size_t row, size_t col);
    const float& at(size_t row, size_t col) const;

    // Data access
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    // Additional operations from components.cpp
    Vector row(size_t row) const;
    void set_row(size_t row, const Vector& vec);
    Matrix transpose() const;
    void apply_relu();
    void apply_gelu();
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

    // Add empty() method
    bool empty() const { return data_.empty(); }
};

class Vector {
private:
    std::vector<float> data_;
    size_t size_;

public:
    // Add default constructor
    Vector() : size_(0) {}
    
    // Existing constructors
    Vector(size_t size, float default_value = 0.0f);
    Vector(const std::initializer_list<float>& list);
    template<typename Iterator>
    Vector(Iterator first, Iterator last) 
        : data_(first, last), size_(std::distance(first, last)) {}
    
    // Data access
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    size_t size() const { return size_; }
    
    // Element access
    float& operator[](size_t i) { return data_[i]; }
    const float& operator[](size_t i) const { return data_[i]; }
    
    // Iterator access
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }
    
    // Serialization
    void save(std::ostream& os) const;
    static Vector load(std::istream& is);

    // Add randomize method
    void randomize(float min_val, float max_val) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(min_val, max_val);
        for (float& val : data_) {
            val = dis(gen);
        }
    }

    // Add resize method
    void resize(size_t new_size) {
        data_.resize(new_size);
        size_ = new_size;
    }
};

// Non-member operators
Matrix operator+(const Matrix& a, const Matrix& b);
Matrix operator-(const Matrix& a, const Matrix& b);
Matrix operator*(const Matrix& m, float scalar);
Matrix operator*(float scalar, const Matrix& m);
Matrix operator/(const Matrix& m, float scalar);
Matrix operator*(const Matrix& a, const Matrix& b);
Matrix matmul(const Matrix& a, const Matrix& b); 