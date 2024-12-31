#pragma once

#include "components.hpp"
#include <unordered_map>

class GradientCheckpoint {
public:
    static void save_activation(const Matrix& activation, size_t layer);
    static Matrix get_activation(size_t layer);

private:
    static std::unordered_map<size_t, Matrix> checkpoints;
}; 