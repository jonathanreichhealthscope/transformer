#pragma once

// Standard library includes
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>

// Project includes
#include "attention.hpp"
#include "lm_head.hpp"
#include "logger.hpp"
#include "matrix.hpp"
#include "model_saver.hpp"
#include "optimizer/sam.hpp"
#include "performance_metrics.hpp"
#include "preprocessing.hpp"
#include "quantization.hpp"
#include "tokenizer.hpp"
#include "transformer.hpp"
#include "utils.hpp"
#include "utils/tensor_cache.hpp"
#include "vocabulary.hpp"

#ifdef CUDA_AVAILABLE
#include "cuda/cuda_init.cuh"
#endif

// Forward declarations
class Tokenizer;
class Matrix;
class Transformer;
class LanguageModelHead;

// Declare global variables as extern
extern std::unique_ptr<Tokenizer> tokenizer;
extern PerformanceMetrics metrics;