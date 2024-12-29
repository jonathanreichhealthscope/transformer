#pragma once
#include "transformer.hpp"
#include <string>

// Save model to file
void save_model(const std::string& path, const Transformer& model);

// Load model from file
void load_model(const std::string& path, Transformer& model); 