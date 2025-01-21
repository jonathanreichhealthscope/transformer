#include "../include/cache.hpp"

KVCache::KVCache(size_t max_len) {
    // Don't create matrices in constructor
    // They will be initialized properly when update() is first called
}

void KVCache::clear() {
    key_cache = Matrix();
    value_cache = Matrix();
}

void KVCache::update(const Matrix& new_keys, const Matrix& new_values) {
    if (key_cache.empty()) {
        // First update - initialize matrices with proper dimensions
        key_cache = new_keys;
        value_cache = new_values;
    } else {
        // Concatenate with existing cache
        Matrix new_key_cache(key_cache.rows() + new_keys.rows(), new_keys.cols());
        Matrix new_value_cache(value_cache.rows() + new_values.rows(), new_values.cols());

        // Copy existing cache
        for (size_t i = 0; i < key_cache.rows(); i++) {
            for (size_t j = 0; j < key_cache.cols(); j++) {
                new_key_cache(i, j) = key_cache(i, j);
                new_value_cache(i, j) = value_cache(i, j);
            }
        }

        // Copy new values
        for (size_t i = 0; i < new_keys.rows(); i++) {
            for (size_t j = 0; j < new_keys.cols(); j++) {
                new_key_cache(i + key_cache.rows(), j) = new_keys(i, j);
                new_value_cache(i + value_cache.rows(), j) = new_values(i, j);
            }
        }

        key_cache = std::move(new_key_cache);
        value_cache = std::move(new_value_cache);
    }
}

std::pair<Matrix, Matrix> KVCache::get_cached_kv() const {
    return {key_cache, value_cache};
}