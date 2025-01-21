#pragma once
#include "transformer.hpp"
#include <string>

/**
 * @file serialization.hpp
 * @brief Model serialization utilities for saving and loading transformer models.
 * 
 * This header provides functionality for persisting transformer models to disk
 * and loading them back. Features include:
 * - Binary serialization format for efficiency
 * - Complete model state preservation
 * - Version compatibility checks
 * - Error handling for I/O operations
 */

/**
 * @brief Saves a transformer model to disk in binary format.
 * 
 * Serializes all model components including:
 * - Model architecture configuration
 * - Layer weights and biases
 * - Embedding tables
 * - Optimizer states (if present)
 * - Tokenizer vocabulary
 * 
 * @param path File path where the model should be saved
 * @param model Transformer model to serialize
 * @throws std::runtime_error if file operations fail
 */
void save_model(const std::string& path, const Transformer& model);

/**
 * @brief Loads a transformer model from disk.
 * 
 * Deserializes a previously saved model, reconstructing:
 * - Model architecture and configuration
 * - All layer parameters
 * - Embedding tables
 * - Optimizer states (if saved)
 * - Tokenizer vocabulary
 * 
 * @param path Path to the saved model file
 * @param model Transformer model to load into
 * @throws std::runtime_error if file operations fail or version mismatch
 */
void load_model(const std::string& path, Transformer& model);