#pragma once
#include "components.hpp"
#include "cache.hpp"
#include <optional>

class AttentionMask {
public:
    Matrix mask;
    
    static AttentionMask create_causal_mask(size_t size);
    static AttentionMask create_padding_mask(const std::vector<int>& lengths, size_t max_len);
};

class MultiHeadAttention {
private:
    Matrix query_proj;
    Matrix key_proj;
    Matrix value_proj;
    Matrix output_proj;
    
    size_t num_heads;
    size_t head_dim;
    bool use_rope;
    bool use_flash;
    bool use_sliding_window;
    size_t window_size;
    
    // RoPE buffers
    Matrix cos_cached;
    Matrix sin_cached;
    
    Matrix apply_rope(const Matrix& x, size_t position) const;
    Matrix flash_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                         const AttentionMask& mask) const;
    Matrix standard_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                            const AttentionMask& mask) const;

public:
    MultiHeadAttention(size_t hidden_size, 
                      size_t num_heads,
                      size_t head_dim,
                      float dropout_prob = 0.1f,
                      bool use_flash = true,
                      bool use_rope = true,
                      bool use_sliding_window = false,
                      size_t window_size = 512,
                      bool use_gqa = false,
                      size_t num_kv_heads = 0);
                      
    Matrix forward(const Matrix& x,
                  const AttentionMask& mask,
                  const std::optional<KVCache>& kv_cache = std::nullopt);
    void save(std::ostream& os) const;
    static std::unique_ptr<MultiHeadAttention> load(std::istream& is);
}; 