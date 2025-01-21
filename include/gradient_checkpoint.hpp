#pragma once

#include "components.hpp"
#include <unordered_map>

class GradientCheckpoint {
  public:
    static void save_activation(const Matrix& activation, size_t layer);
    static Matrix get_activation(size_t layer);
    static void clear_cache();
    static void cache_activation(const std::string& key, const Matrix& activation);
    static Matrix get_activation(const std::string& key);
    static bool has_activation(const std::string& key);

  private:
    static std::unordered_map<size_t, Matrix> checkpoints;
    static std::unordered_map<std::string, Matrix> activation_cache;
};