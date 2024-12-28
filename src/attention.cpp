#include "../include/attention.hpp"
#include <cmath>

Matrix MultiHeadAttention::apply_rope(const Matrix& x, size_t position) const {
    Matrix rotated = x;
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < x.rows(); ++i) {
            for (size_t j = 0; j < head_dim; j += 2) {
                float cos_theta = cos_cached[position][j/2];
                float sin_theta = sin_cached[position][j/2];
                
                size_t idx = h * head_dim + j;
                float x1 = x[i][idx];
                float x2 = x[i][idx + 1];
                
                rotated[i][idx] = x1 * cos_theta - x2 * sin_theta;
                rotated[i][idx + 1] = x1 * sin_theta + x2 * cos_theta;
            }
        }
    }
    return rotated;
}

Matrix MultiHeadAttention::flash_attention(const Matrix& Q, const Matrix& K, 
                                         const Matrix& V, const AttentionMask& mask) const {
    // Implement Flash Attention algorithm
    // This is a simplified version - real implementation would use block-sparse operations
    const size_t block_size = 256;  // Typical block size for GPU
    Matrix output(Q.rows(), V.cols(), 0.0f);
    
    for (size_t b_start = 0; b_start < K.rows(); b_start += block_size) {
        size_t b_end = std::min(b_start + block_size, K.rows());
        
        // Load K,V blocks
        Matrix K_block = K.block(b_start, b_end);
        Matrix V_block = V.block(b_start, b_end);
        
        // Compute attention scores for block
        Matrix scores = Matrix::matmul(Q, K_block.transpose());
        scores *= 1.0f / std::sqrt(head_dim);
        
        if (mask.mask.rows() > 0) {
            // Apply attention mask
            scores *= mask.mask;
        }
        
        scores.apply_softmax();
        
        // Accumulate output
        output += Matrix::matmul(scores, V_block);
    }
    
    return output;
}

Matrix MultiHeadAttention::forward(const Matrix& x,
                                 const AttentionMask& mask,
                                 const std::optional<KVCache>& kv_cache) {
    // Project inputs
    Matrix Q = Matrix::matmul(x, query_proj);
    Matrix K = Matrix::matmul(x, key_proj);
    Matrix V = Matrix::matmul(x, value_proj);
    
    // Apply RoPE if enabled
    if (use_rope) {
        for (size_t pos = 0; pos < x.rows(); ++pos) {
            Q.row(pos) = apply_rope(Q.row(pos), pos);
            K.row(pos) = apply_rope(K.row(pos), pos);
        }
    }
    
    // Use cached KV if provided
    if (kv_cache) {
        auto [cached_k, cached_v] = kv_cache->get_cached_kv();
        K = Matrix::concatenate(cached_k, K);
        V = Matrix::concatenate(cached_v, V);
    }
    
    // Compute attention
    Matrix attention_output;
    if (use_flash) {
        attention_output = flash_attention(Q, K, V, mask);
    } else {
        attention_output = standard_attention(Q, K, V, mask);
    }
    
    // Project output
    return Matrix::matmul(attention_output, output_proj);
}

void MultiHeadAttention::save(std::ostream& os) const {
    // Save dimensions and configuration
    os.write(reinterpret_cast<const char*>(&num_heads), sizeof(num_heads));
    os.write(reinterpret_cast<const char*>(&head_dim), sizeof(head_dim));
    
    // Save projection matrices
    query_proj.save(os);
    key_proj.save(os);
    value_proj.save(os);
    output_proj.save(os);
}

std::unique_ptr<MultiHeadAttention> MultiHeadAttention::load(std::istream& is) {
    size_t num_heads, head_dim;
    is.read(reinterpret_cast<char*>(&num_heads), sizeof(num_heads));
    is.read(reinterpret_cast<char*>(&head_dim), sizeof(head_dim));
    
    auto attention = std::make_unique<MultiHeadAttention>(
        num_heads * head_dim,  // hidden_size
        num_heads,
        head_dim
    );
    
    // Load projection matrices
    attention->query_proj = Matrix::load(is);
    attention->key_proj = Matrix::load(is);
    attention->value_proj = Matrix::load(is);
    attention->output_proj = Matrix::load(is);
    
    return attention;
} 