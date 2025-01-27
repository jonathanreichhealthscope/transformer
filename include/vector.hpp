#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <iostream>

// Forward declare Matrix if needed
class Matrix;

class Vector {
private:
    std::vector<float> data_;
    size_t size_;

public:
    // Constructors
    Vector();
    Vector(size_t size, float default_value = 0.0f);
    Vector(const std::initializer_list<float>& list);
    
    template <typename Iterator>
    Vector(Iterator first, Iterator last) : data_(first, last), size_(std::distance(first, last)) {}

    // Data access
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    size_t size() const { return size_; }

    // Element access
    float& operator[](size_t i) { return data_[i]; }
    const float& operator[](size_t i) const { return data_[i]; }

    // Operators
    Vector& operator+=(const Vector& other);

    // Iterator access
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }

    // Utility functions
    bool empty() const { return data_.empty(); }
    void resize(size_t new_size);
    void fill(float value);
    void randomize(float min_val, float max_val);

    // Initialization methods
    void initialize_random(float scale);
    void initialize_constant(float value);

    // Serialization
    void save(std::ostream& os) const;
    static Vector load(std::istream& is);
};

// Non-member operators
Vector operator+(const Vector& a, const Vector& b);
Vector operator-(const Vector& a, const Vector& b);
Vector operator*(const Vector& v, float scalar);
Vector operator*(float scalar, const Vector& v); 