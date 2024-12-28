#include "../include/transformer.hpp"
#include <stdexcept>
#include <fstream>

// TransformerConfig implementation
TransformerConfig::TransformerConfig(
    size_t vocab_size,
    size_t max_seq_length,
    size_t hidden_size,
    size_t num_layers,
    size_t num_heads
) : vocab_size(vocab_size),
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
    num_kv_heads(num_heads),
    use_cuda(true)
{
    if (hidden_size % num_heads != 0) {
        throw std::invalid_argument("Hidden size must be divisible by number of heads");
    }
}

// TransformerLayer implementation
TransformerLayer::TransformerLayer(const TransformerConfig& config) 
    : kv_cache(config.max_seq_length),
      config(config)
{
    // Initialize attention layer
    self_attention = std::make_unique<MultiHeadAttention>(
        config.hidden_size,
        config.num_heads,
        config.head_dim,
        config.dropout_prob,
        config.use_flash_attention,
        config.use_rope,
        config.use_sliding_window,
        config.window_size,
        config.use_gqa,
        config.num_kv_heads
    );
    
    // Initialize layer normalization
    attention_ln = std::make_unique<LayerNorm>(config.hidden_size);
    ffn_ln = std::make_unique<LayerNorm>(config.hidden_size);
    
    // Initialize feed-forward network
    feed_forward = std::make_unique<FeedForward>(
        config.hidden_size,
        config.intermediate_size,
        config.dropout_prob
    );
}

Matrix TransformerLayer::forward(const Matrix& x, const AttentionMask& mask) {
    // Pre-layer normalization
    Matrix normalized = attention_ln->forward(x);
    
    // Self-attention with residual connection
    Matrix attention_output = self_attention->forward(normalized, mask, std::make_optional(std::ref(kv_cache)));
    Matrix residual = x + attention_output;
    
    // Feed-forward with residual connection
    normalized = ffn_ln->forward(residual);
    Matrix ffn_output = feed_forward->forward(normalized);
    
    return residual + ffn_output;
}

void TransformerLayer::clear_cache() {
    kv_cache.clear();
}

void TransformerLayer::save(std::ostream& os) const {
    self_attention->save(os);
    attention_ln->save(os);
    feed_forward->save(os);
    ffn_ln->save(os);
}

std::unique_ptr<TransformerLayer> TransformerLayer::load(std::istream& is) {
    auto config = TransformerConfig(); // You'll need to load this or pass it
    auto layer = std::make_unique<TransformerLayer>(config);
    layer->self_attention = MultiHeadAttention::load(is);
    layer->attention_ln = LayerNorm::load(is);
    layer->feed_forward = FeedForward::load(is);
    layer->ffn_ln = LayerNorm::load(is);
    return layer;
}

// Transformer implementation
Transformer::Transformer(const TransformerConfig& config) : config(config) {
    // Initialize token embedding
    token_embedding = std::make_unique<TokenEmbedding>(
        config.vocab_size,
        config.hidden_size
    );
    
    // Initialize positional encoding
    pos_encoding = std::make_unique<PositionalEncoding>(
        config.max_seq_length,
        config.hidden_size
    );
    
    // Initialize transformer layers
    layers.reserve(config.num_layers);
    for (size_t i = 0; i < config.num_layers; ++i) {
        layers.push_back(std::make_unique<TransformerLayer>(config));
    }
    
    // Initialize final layer normalization
    final_ln = std::make_unique<LayerNorm>(config.hidden_size);
    
#ifdef USE_CUDA
    if (config.use_cuda) {
        cuda_manager = std::make_unique<CudaManager>();
    }
#endif
}

Matrix Transformer::forward(
    const std::vector<int>& input_tokens,
    bool use_cache
) {
    // Get embeddings
    Matrix embeddings = token_embedding->forward(input_tokens);
    
    // Add positional encodings
    Matrix position_ids(1, input_tokens.size());
    for (size_t i = 0; i < input_tokens.size(); ++i) {
        position_ids(0, i) = static_cast<float>(i);
    }
    embeddings += pos_encoding->forward(position_ids);
    
    // Create attention mask if needed
    AttentionMask mask;
    if (!use_cache) {
        mask = AttentionMask::create_causal_mask(input_tokens.size());
    }
    
    // Forward pass through layers
    Matrix hidden_states = embeddings;
    for (auto& layer : layers) {
        hidden_states = layer->forward(hidden_states, mask);
    }
    
    // Final layer normalization
    hidden_states = final_ln->forward(hidden_states);
    
    // Project to vocabulary (reuse embedding weights)
    return token_embedding->project_to_vocab(hidden_states);
}

void Transformer::train(
    const std::vector<std::vector<int>>& input_tokens,
    const std::vector<std::vector<int>>& target_tokens,
    size_t num_epochs,
    float learning_rate
) {
    TransformerTrainer trainer(*this, learning_rate, 32);
    
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        // Process batches
        for (size_t i = 0; i < input_tokens.size(); i += trainer.batch_size()) {
            size_t batch_end = std::min(i + trainer.batch_size(), input_tokens.size());
            std::vector<std::vector<int>> input_batch(
                input_tokens.begin() + i,
                input_tokens.begin() + batch_end
            );
            std::vector<std::vector<int>> target_batch(
                target_tokens.begin() + i,
                target_tokens.begin() + batch_end
            );
            
            trainer.train_step(input_batch, target_batch);
        }
    }
}

void Transformer::save_model(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) {
        throw std::runtime_error("Failed to open file for saving model");
    }
    
    // Save config
    os.write(reinterpret_cast<const char*>(&config), sizeof(config));
    
    // Save embeddings
    token_embedding->save(os);
    pos_encoding->save(os);
    
    // Save layers
    for (const auto& layer : layers) {
        layer->save(os);
    }
    
    // Save final layer norm
    final_ln->save(os);
}

Transformer Transformer::load_model(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        throw std::runtime_error("Failed to open file for loading model");
    }
    
    // Load config
    TransformerConfig config;
    is.read(reinterpret_cast<char*>(&config), sizeof(config));
    
    // Create transformer with loaded config
    Transformer transformer(config);
    
    // Load embeddings
    transformer.token_embedding = std::move(TokenEmbedding::load(is));
    transformer.pos_encoding = std::move(PositionalEncoding::load(is));
    
    // Load layers
    transformer.layers.clear();
    for (size_t i = 0; i < config.num_layers; ++i) {
        transformer.layers.push_back(TransformerLayer::load(is));
    }
    
    // Load final layer norm
    transformer.final_ln = std::move(LayerNorm::load(is));
    
    return transformer;
}

void Transformer::clear_kv_cache() {
    for (auto& layer : layers) {
        layer->clear_cache();
    }
} 