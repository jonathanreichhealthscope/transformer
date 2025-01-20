#pragma once

// Standard library includes
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <sstream>

// Project includes
#include "attention.hpp"
#include "beam_search.hpp"
#include "lm_head.hpp"
#include "logger.hpp"
#include "model_saver.hpp"
#include "optimizer/sam.hpp"
#include "performance_metrics.hpp"
#include "quantization.hpp"
#include "tokenizer.hpp"
#include "transformer.hpp"
#include "utils.hpp"
#include "utils/tensor_cache.hpp"
#include "vocabulary.hpp"
#include "matrix.hpp"
#include "preprocessing.hpp"

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