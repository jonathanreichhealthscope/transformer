#include "../include/cache.hpp"

KVCache::KVCache(size_t max_length) : max_seq_length(max_length) {
    clear();
}

void KVCache::update(const Matrix& new_keys, const Matrix& new_values) {
    // Add new keys and values to the cache
    key_cache.push_back(new_keys);
    value_cache.push_back(new_values);
    
    // Ensure we don't exceed max sequence length
    while (key_cache.size() > max_seq_length) {
        key_cache.pop_front();
        value_cache.pop_front();
    }
}

std::pair<Matrix, Matrix> KVCache::get_cached_kv() const {
    if (key_cache.empty() || value_cache.empty()) {
        return {Matrix(), Matrix()};
    }
    
    // Calculate total number of tokens in cache
    size_t total_tokens = 0;
    for (const auto& k : key_cache) {
        total_tokens += k.rows();
    }
    
    // Get dimensions from first cached matrices
    const size_t head_dim = key_cache.front().cols();
    const size_t batch_size = 1;  // Assuming batch size of 1 for inference
    
    // Allocate concatenated matrices
    Matrix concatenated_keys(total_tokens, head_dim);
    Matrix concatenated_values(total_tokens, head_dim);
    
    // Copy cached keys and values into concatenated matrices
    size_t current_row = 0;
    for (size_t i = 0; i < key_cache.size(); ++i) {
        const auto& k = key_cache[i];
        const auto& v = value_cache[i];
        
        // Copy key rows
        for (size_t row = 0; row < k.rows(); ++row) {
            for (size_t col = 0; col < head_dim; ++col) {
                concatenated_keys(current_row + row, col) = k(row, col);
                concatenated_values(current_row + row, col) = v(row, col);
            }
        }
        
        current_row += k.rows();
    }
    
    return {concatenated_keys, concatenated_values};
}

void KVCache::clear() {
    key_cache.clear();
    value_cache.clear();
} 