#pragma once
#include "transformer.hpp"
#include <string>

// Simple binary serialization functions
void save_model(const std::string& path, const Transformer& model);
void load_model(const std::string& path, Transformer& model); 