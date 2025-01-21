#pragma once
#include "matrix.hpp"

/**
 * @file types.hpp
 * @brief Common type definitions used throughout the transformer implementation.
 * 
 * This header provides type aliases and common definitions to ensure
 * consistency across the codebase and make it easier to modify underlying
 * types if needed. Features include:
 * - Vector type aliases
 * - Common mathematical types
 * - Platform-specific type definitions
 */

/**
 * @brief Alias for floating-point vector type.
 * 
 * Uses the Vector class from matrix.hpp as the underlying implementation.
 * This alias allows for easy switching between different vector implementations
 * or precision levels (e.g., float vs double) if needed.
 */
using FloatVector = Vector;