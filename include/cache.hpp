#pragma once
#include "matrix.hpp"

/**
 * @brief Key-Value cache for efficient autoregressive generation.
 * 
 * The KVCache struct implements caching of key and value tensors in transformer
 * attention layers, enabling efficient autoregressive generation by avoiding
 * redundant computation of keys and values for previously processed tokens.
 * Features include:
 * - Efficient memory management
 * - Dynamic cache updates
 * - Automatic size handling
 * - Cache invalidation
 */
struct KVCache {
    Matrix key_cache;    ///< Cached key tensors from previous positions
    Matrix value_cache;  ///< Cached value tensors from previous positions

    /**
     * @brief Constructs a key-value cache.
     * @param max_len Maximum sequence length to cache (0 for dynamic sizing)
     */
    explicit KVCache(size_t max_len = 0);

    /**
     * @brief Clears all cached keys and values.
     * 
     * Should be called when starting generation for a new sequence
     * or when cache becomes invalid.
     */
    void clear();

    /**
     * @brief Checks if the cache is empty.
     * @return true if either key or value cache is empty
     */
    bool empty() const {
        return key_cache.empty() || value_cache.empty();
    }

    /**
     * @brief Updates the cache with new key-value pairs.
     * 
     * Appends new keys and values to the cache, extending its
     * size if necessary. This is typically called during
     * autoregressive generation after computing attention for
     * new tokens.
     * 
     * @param new_keys New key tensors to cache
     * @param new_values New value tensors to cache
     */
    void update(const Matrix& new_keys, const Matrix& new_values);

    /**
     * @brief Retrieves all cached key-value pairs.
     * 
     * Returns the complete history of cached keys and values,
     * which can be used to compute attention scores for new
     * tokens against all previous positions.
     * 
     * @return Pair of matrices containing all cached keys and values
     */
    std::pair<Matrix, Matrix> get_cached_kv() const;
};