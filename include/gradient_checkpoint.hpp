#pragma once

#include "components.hpp"
#include <unordered_map>

/**
 * @brief Implements gradient checkpointing for memory-efficient training.
 * 
 * Gradient checkpointing is a technique that trades computation for memory
 * by storing only selected layer activations during the forward pass and
 * recomputing others during backpropagation. Features include:
 * - Layer-wise activation storage
 * - Key-based activation caching
 * - Memory-efficient backpropagation
 * - Configurable checkpointing strategy
 * 
 * This implementation helps train large models that would otherwise exceed
 * available memory by reducing the memory required to store activations.
 */
class GradientCheckpoint {
  public:
    /**
     * @brief Saves an activation at a specific layer.
     * 
     * During the forward pass, this method stores activations that will be
     * needed during backpropagation.
     * 
     * @param activation Activation tensor to save
     * @param layer Layer index to associate with the activation
     */
    static void save_activation(const Matrix& activation, size_t layer);

    /**
     * @brief Retrieves an activation from a specific layer.
     * 
     * Used during backpropagation to access stored activations
     * needed for gradient computation.
     * 
     * @param layer Layer index to retrieve activation from
     * @return Stored activation matrix
     */
    static Matrix get_activation(size_t layer);

    /**
     * @brief Clears all stored checkpoints and cached activations.
     * 
     * Should be called at the end of each backward pass to free memory.
     */
    static void clear_cache();

    /**
     * @brief Caches an activation with a string key.
     * 
     * Provides an alternative to layer-based storage for cases where
     * a more flexible naming scheme is needed.
     * 
     * @param key String identifier for the activation
     * @param activation Activation tensor to cache
     */
    static void cache_activation(const std::string& key, const Matrix& activation);

    /**
     * @brief Retrieves a cached activation by key.
     * 
     * @param key String identifier of the activation to retrieve
     * @return Cached activation matrix
     */
    static Matrix get_activation(const std::string& key);

    /**
     * @brief Checks if an activation exists in the cache.
     * 
     * @param key String identifier to check
     * @return true if the activation exists in cache, false otherwise
     */
    static bool has_activation(const std::string& key);

  private:
    static std::unordered_map<size_t, Matrix> checkpoints;      ///< Layer-indexed activation storage
    static std::unordered_map<std::string, Matrix> activation_cache; ///< Key-value activation storage
};