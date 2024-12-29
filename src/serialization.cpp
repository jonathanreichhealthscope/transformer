#include "../include/serialization.hpp"
#include <fstream>

void save_model(const std::string &path, const Transformer &model) {
  std::ofstream os(path, std::ios::binary);
  if (!os) {
    throw std::runtime_error("Failed to open file for saving");
  }
  model.save(os);
}

void load_model(const std::string &path, Transformer &model) {
  std::ifstream is(path, std::ios::binary);
  if (!is) {
    throw std::runtime_error("Failed to open file for loading");
  }
  model.load(is);
}