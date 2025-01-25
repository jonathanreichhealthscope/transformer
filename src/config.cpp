#include "../include/config.hpp"
#include <iostream>
#include <stdexcept>

TransformerConfig::TransformerConfig(size_t vocab_size, size_t max_seq_length, size_t hidden_size,
                                     size_t num_layers, size_t num_heads, size_t batch_size,
                                     size_t num_epochs)
    : vocab_size(vocab_size), max_seq_length(max_seq_length), hidden_size(hidden_size),
      num_layers(num_layers), num_heads(num_heads), head_dim(hidden_size / num_heads),
      intermediate_size(4 * hidden_size), dropout_prob(0.1f), use_flash_attention(true),
      use_rope(true), use_sliding_window(false), window_size(512), use_gqa(false),
      num_kv_heads(num_heads / 2), use_fp16(false), use_gradient_checkpointing(true),
      memory_pool_size(1024), batch_size(batch_size), num_epochs(num_epochs), dropout_rate(0.1f),
      weight_decay(0.01f), load_from_checkpoint(false), checkpoint_to_load(""),
      paths{
          "models",            // save_directory
          "transformer_model", // model_name
          2                    // checkpoint_frequency
      },
      // Initialize beam search parameters with defaults
      beam_search{
          true,  // use_beam_search
          5,     // beam_size
          0.6f,  // length_penalty
          1.0f,  // temperature
          3.0f,  // initial_temperature
          4.0f,  // diversity_strength
          100,   // top_k
          0.9f,  // top_p
          20,    // max_length
          0.8f,  // initial_noise_scale
          0.1f   // token_noise_scale
      } {
    std::cout << "entering TransformerConfig constructor" << std::endl;
    if (hidden_size % num_heads != 0) {
        throw std::invalid_argument("Hidden size must be divisible by number of heads");
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
           head_dim != other.head_dim ||
           intermediate_size != other.intermediate_size ||
           batch_size != other.batch_size ||
           num_epochs != other.num_epochs ||
           beam_search.beam_size != other.beam_search.beam_size ||
           beam_search.length_penalty != other.beam_search.length_penalty ||
           beam_search.temperature != other.beam_search.temperature ||
           beam_search.top_p != other.beam_search.top_p ||
           beam_search.max_length != other.beam_search.max_length;
}