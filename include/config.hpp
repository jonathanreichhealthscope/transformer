#pragma once
#include <string>
#include <cstddef>

class TransformerConfig {
public:
    // Model parameters
    size_t vocab_size;
    size_t max_seq_length;
    size_t hidden_size;
    size_t num_layers;
    size_t num_heads;
    size_t head_dim;
    size_t intermediate_size;
    float dropout_prob;

    // Attention parameters
    bool use_flash_attention;
    bool use_rope;
    bool use_sliding_window;
    size_t window_size;
    bool use_gqa;
    size_t num_kv_heads;

    // Optimization parameters
    bool use_fp16;
    bool use_gradient_checkpointing;
    size_t memory_pool_size;
    size_t batch_size;
    size_t num_epochs;
    float dropout_rate;
    float weight_decay;

    // Path settings
    struct {
        std::string save_directory;
        std::string model_name;
        size_t checkpoint_frequency;
    } paths;

    // Checkpoint settings
    bool load_from_checkpoint;
    std::string checkpoint_to_load;

    // Beam search parameters
    size_t beam_size;
    float length_penalty;
    float temperature;
    float top_p;
    size_t max_length;

    // Constructor
    TransformerConfig(size_t vocab_size = 32000, 
                     size_t max_seq_length = 512,
                     size_t hidden_size = 768, 
                     size_t num_layers = 12,
                     size_t num_heads = 12, 
                     size_t batch_size = 32,
                     size_t num_epochs = 10);

    // Make operator!= a member function instead
    bool operator!=(const TransformerConfig& other) const;
}; 