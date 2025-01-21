#pragma once
#include "matrix.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>

/**
 * @file components.hpp
 * @brief Common components and utilities for transformer model implementation.
 * 
 * This header provides shared functionality used across the transformer
 * implementation, including:
 * - Matrix and Vector type definitions
 * - Common mathematical operations
 * - Utility functions for debugging and visualization
 * - Memory management utilities
 * 
 * The components in this file serve as building blocks for more complex
 * transformer layers and operations.
 */

// Use Matrix and Vector from matrix.hpp
// Rest of components.hpp content...

/**
 * @brief Prints statistical information about a matrix.
 * 
 * Outputs useful debugging information about a matrix, including:
 * - Dimensions
 * - Value range (min/max)
 * - Mean and standard deviation
 * - Number of non-zero elements
 * - Memory usage
 * 
 * @param m Matrix to analyze
 */
void print_matrix_stats(const Matrix& m);