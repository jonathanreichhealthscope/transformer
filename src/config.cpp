#include "../include/config.hpp"
#include <iostream>
#include <stdexcept>

TransformerConfig::TransformerConfig(size_t vocab_size, size_t max_seq_length,
                                   size_t hidden_size, size_t num_layers,
                                   size_t num_heads, size_t batch_size,
                                   size_t num_epochs)
    : vocab_size(vocab_size), 
      max_seq_length(max_seq_length),
      hidden_size(hidden_size), 
      num_layers(num_layers), 
      num_heads(num_heads),
      head_dim(hidden_size / num_heads), 
      intermediate_size(4 * hidden_size),
      dropout_prob(0.1f), 
      use_flash_attention(true), 
      use_rope(true),
      use_sliding_window(false), 
      window_size(512), 
      use_gqa(false),
      num_kv_heads(num_heads/2),
      use_fp16(false), 
      use_gradient_checkpointing(true),
      memory_pool_size(1024), 
      batch_size(batch_size),
      num_epochs(num_epochs),
      dropout_rate(0.1f),
      weight_decay(0.01f),
      load_from_checkpoint(false),
      checkpoint_to_load(""),
      paths{
          "models",           // save_directory
          "transformer_model", // model_name
          2                   // checkpoint_frequency
      },
      // Initialize beam search parameters with defaults
      beam_size(5),
      length_penalty(0.6f),
      temperature(1.0f),
      top_p(0.9f),
      max_length(20)
{
    std::cout << "entering TransformerConfig constructor" << std::endl;
    if (hidden_size % num_heads != 0) {
        throw std::invalid_argument(
            "Hidden size must be divisible by number of heads");
    }
    std::cout << "exiting TransformerConfig constructor" << std::endl;
}

// Define as a member function
bool TransformerConfig::operator!=(const TransformerConfig& other) const {
    return vocab_size != other.vocab_size ||
           max_seq_length != other.max_seq_length ||
           hidden_size != other.hidden_size ||
           num_layers != other.num_layers || 
           num_heads != other.num_heads ||
           use_flash_attention != other.use_flash_attention ||
           use_rope != other.use_rope ||
           use_sliding_window != other.use_sliding_window ||
           window_size != other.window_size ||
           batch_size != other.batch_size ||
           num_epochs != other.num_epochs ||
           beam_size != other.beam_size ||
           length_penalty != other.length_penalty ||
           temperature != other.temperature ||
           top_p != other.top_p ||
           max_length != other.max_length;
} 