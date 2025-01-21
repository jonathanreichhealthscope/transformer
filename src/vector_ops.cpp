#include "../include/matrix.hpp"

Vector::Vector() : size_(0) {}

Vector::Vector(size_t size, float default_value) : data_(size, default_value), size_(size) {}

Vector::Vector(const std::initializer_list<float>& list) : data_(list), size_(list.size()) {}

void Vector::save(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&size_), sizeof(size_));
    os.write(reinterpret_cast<const char*>(data_.data()), data_.size() * sizeof(float));
}

Vector Vector::load(std::istream& is) {
    size_t size;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    Vector result(size);
    is.read(reinterpret_cast<char*>(result.data_.data()), size * sizeof(float));
    return result;
}