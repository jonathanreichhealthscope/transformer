#include "../include/components.hpp"

Vector::Vector(size_t size, float default_value)
    : data_(size, default_value), size_(size) {}

Vector::Vector(const std::initializer_list<float>& list)
    : data_(list), size_(list.size()) {} 