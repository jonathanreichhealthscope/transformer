#pragma once
#include <cstddef>
#include <string>
#include <vector>

/**
 * @brief Configuration class for transformer model architecture and training settings.
 * 
 * This class encapsulates all configuration parameters needed to define and train
 * a transformer model, including architectural choices, optimization settings,
 * and various hyperparameters. The configuration is divided into several categories:
 * - Model architecture parameters
 * - Attention mechanism settings
 * - Training and optimization parameters
 * - File paths and checkpointing
 * - Generation and beam search settings
 */
class TransformerConfig {
  public:
    // Model parameters
    size_t vocab_size;           ///< Size of the vocabulary
    size_t max_seq_length;       ///< Maximum sequence length the model can handle
    size_t hidden_size;          ///< Dimension of the model's hidden states
    size_t num_layers;           ///< Number of transformer layers
    size_t num_heads;            ///< Number of attention heads
    size_t head_dim;             ///< Dimension of each attention head
    size_t intermediate_size;     ///< Size of the feedforward network
    float dropout_prob;          ///< Probability of dropout

    // Attention parameters
    bool use_flash_attention;     ///< Whether to use flash attention optimization
    bool use_rope;               ///< Whether to use rotary positional embeddings
    bool use_sliding_window;     ///< Whether to use sliding window attention
    size_t window_size;          ///< Size of the attention window if using sliding window
    bool use_gqa;               ///< Whether to use grouped-query attention
    size_t num_kv_heads;         ///< Number of key/value heads for GQA

    // Optimization parameters
    bool use_fp16;               ///< Whether to use half-precision training
    bool use_gradient_checkpointing; ///< Whether to use gradient checkpointing
    size_t memory_pool_size;     ///< Size of memory pool for optimizations
    size_t batch_size;           ///< Training batch size
    size_t num_epochs;           ///< Number of training epochs
    float dropout_rate;          ///< Global dropout rate
    float weight_decay;          ///< L2 regularization factor

    // Path settings
    struct {
        std::string save_directory;      ///< Directory to save model checkpoints
        std::string model_name;          ///< Name of the model
        size_t checkpoint_frequency;      ///< How often to save checkpoints
    } paths;

    // Checkpoint settings
    bool load_from_checkpoint;    ///< Whether to resume from checkpoint
    std::string checkpoint_to_load; ///< Path to checkpoint file to load

    // Beam search parameters
    struct BeamSearchConfig {
        bool use_beam_search = true;
        size_t beam_size = 5;
        size_t beams_per_group = 4;
        size_t num_groups = 3;
        float length_penalty = 1.5f;
        float temperature = 1.0f;
        float top_p = 0.9f;
        size_t max_length = 20;
        float initial_temperature = 3.0f;
        float initial_noise_scale = 0.8f;
        float diversity_strength = 4.0f;
        size_t top_k = 100;
        float token_noise_scale = 0.1f;
    } beam_search;

    // Tokenizer configuration
    struct TokenizerConfig {
        bool use_subword = true;
        size_t vocab_size = 32000;  // Changed from max_vocab_size
        std::string model_path = "model/tokenizer.model";
        std::vector<std::string> special_tokens = {
            "<pad>", "<unk>", "<bos>", "<eos>", "<mask>"
        };
    } tokenizer;

    /**
     * @brief Constructs a transformer configuration with default values.
     * @param vocab_size Size of the vocabulary (default: 32000)
     * @param max_seq_length Maximum sequence length (default: 512)
     * @param hidden_size Dimension of hidden states (default: 768)
     * @param num_layers Number of transformer layers (default: 12)
     * @param num_heads Number of attention heads (default: 12)
     * @param batch_size Training batch size (default: 32)
     * @param num_epochs Number of training epochs (default: 10)
     */
    TransformerConfig(size_t vocab_size = 32000, size_t max_seq_length = 512,
                      size_t hidden_size = 768, size_t num_layers = 12, size_t num_heads = 12,
                      size_t batch_size = 32, size_t num_epochs = 10);

    /**
     * @brief Compares two configurations for inequality.
     * @param other Configuration to compare against
     * @return true if configurations differ, false otherwise
     */
    bool operator!=(const TransformerConfig& other) const;
};