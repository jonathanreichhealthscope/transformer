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

void Transformer::train(const std::vector<std::vector<int>>& input_tokens,
                       const std::vector<std::vector<int>>& target_tokens,
                       size_t num_epochs,
                       float learning_rate) {
    const size_t batch_size = 32;  // Fixed batch size
    
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        // Process batches
        for (size_t i = 0; i < input_tokens.size(); i += batch_size) {
            size_t batch_end = std::min(i + batch_size, input_tokens.size());
            
            // Create batch
            std::vector<std::vector<int>> input_batch(
                input_tokens.begin() + i,
                input_tokens.begin() + batch_end
            );
            std::vector<std::vector<int>> target_batch(
                target_tokens.begin() + i,
                target_tokens.begin() + batch_end
            );
            
            // Forward pass
            std::vector<Matrix> activations;
            Matrix logits = forward(input_batch[0], &activations);
            
            // Compute loss and gradients
            Matrix loss_grad = compute_loss_gradients(logits, target_batch[0]);
            
            // Backward pass
            backward_pass(activations, loss_grad);
            
            // Update parameters using optimizer (you'll need to implement this)
            update_parameters(learning_rate);
        }
    }
}

Matrix Transformer::compute_loss_gradients(const Matrix& logits, const std::vector<int>& targets) {
    const size_t batch_size = logits.rows();
    const size_t vocab_size = logits.cols();
    Matrix gradients(batch_size, vocab_size);
    
    // For each sequence position
    for (size_t i = 0; i < batch_size; ++i) {
        // Compute softmax probabilities
        std::vector<float> probs(vocab_size);
        float max_logit = -std::numeric_limits<float>::infinity();
        
        // Find max logit for numerical stability
        for (size_t j = 0; j < vocab_size; ++j) {
            max_logit = std::max(max_logit, logits(i, j));
        }
        
        // Compute softmax denominator
        float sum_exp = 0.0f;
        for (size_t j = 0; j < vocab_size; ++j) {
            probs[j] = std::exp(logits(i, j) - max_logit);
            sum_exp += probs[j];
        }
        
        // Normalize and compute gradients
        for (size_t j = 0; j < vocab_size; ++j) {
            probs[j] /= sum_exp;
            // Gradient is (probability - 1) for correct class, probability for others
            gradients(i, j) = probs[j];
        }
        gradients(i, targets[i]) -= 1.0f;  // Subtract 1 from target class
    }
    
    return gradients;
}

void Transformer::backward_pass(const std::vector<Matrix>& activations, const Matrix& loss_grad) {
    Matrix current_grad = loss_grad;
    
    // Backward through final layer norm
    current_grad = final_ln->backward(current_grad, activations.back());
    
    // Backward through layers in reverse order
    for (int i = layers.size() - 1; i >= 0; --i) {
        if (config.use_cuda) {
            current_grad = layers[i]->backward_cuda(current_grad, activations[i]);
        } else {
            current_grad = layers[i]->backward(current_grad, activations[i]);
        }
    }
    
    // Backward through embeddings
    Matrix embedding_grad = current_grad;
    // token_embedding->accumulate_gradients(embedding_grad);
}

void Transformer::update_parameters(float learning_rate) {
    // Get all trainable parameters and their gradients
    auto& params = this->parameters();
    
    // Simple SGD update
    for (size_t i = 0; i < params.size(); ++i) {
        Matrix& param = params[i];
        // Assuming you have stored gradients somewhere
        // Matrix& grad = parameter_gradients[i];
        
        // Update rule: param = param - learning_rate * grad
        for (size_t row = 0; row < param.rows(); ++row) {
            for (size_t col = 0; col < param.cols(); ++col) {
                // param(row, col) -= learning_rate * grad(row, col);
                // For now, just add a placeholder update
                param(row, col) -= learning_rate * 0.01f;  // Replace with actual gradient
            }
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

Matrix Transformer::backward(const Matrix& grad, const Matrix& activation, size_t layer_idx) {
    if (layer_idx >= layers.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    
    // Compute gradients for the current layer
    Matrix layer_grad = grad;
    
    // Backward through layer normalization
    if (layer_idx == layers.size() - 1) {
        layer_grad = final_ln->backward(layer_grad, activation);
    }
    
    // Backward through transformer layer
    // Note: This would require implementing backward methods in TransformerLayer
    // and its components (attention, feed-forward, etc.)
    
    return layer_grad;
}

Matrix Transformer::backward_cuda(const Matrix& grad, const Matrix& activation, size_t layer_idx) {
#ifdef USE_CUDA
    if (!cuda_manager) {
        throw std::runtime_error("CUDA manager not initialized");
    }
    
    if (layer_idx >= layers.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    
    // Similar to CPU version but using CUDA operations
    Matrix layer_grad = grad;
    
    if (layer_idx == layers.size() - 1) {
        layer_grad = final_ln->forward_cuda(layer_grad);
    }
    
    // CUDA backward pass through transformer layer
    
    return layer_grad;
#else
    throw std::runtime_error("CUDA support not enabled");
#endif
}

std::vector<Matrix>& Transformer::parameters() {
    static std::vector<Matrix> all_params;
    all_params.clear();
    
    // Add embedding parameters
    all_params.push_back(token_embedding->weights);
    
    // Add layer parameters
    for (auto& layer : layers) {
        // Add attention parameters
        all_params.push_back(layer->self_attention->query_proj);
        all_params.push_back(layer->self_attention->key_proj);
        all_params.push_back(layer->self_attention->value_proj);
        all_params.push_back(layer->self_attention->output_proj);
        
        // Add layer norm parameters - convert Vector to Matrix
        Matrix gamma_matrix(1, layer->attention_ln->gamma.size());
        Matrix beta_matrix(1, layer->attention_ln->beta.size());
        for (size_t i = 0; i < layer->attention_ln->gamma.size(); ++i) {
            gamma_matrix(0, i) = layer->attention_ln->gamma[i];
            beta_matrix(0, i) = layer->attention_ln->beta[i];
        }
        all_params.push_back(gamma_matrix);
        all_params.push_back(beta_matrix);
        
        // Add feed forward parameters
        all_params.push_back(layer->feed_forward->w1);
        all_params.push_back(layer->feed_forward->w2);
        
        // Convert feed forward biases to matrices
        Matrix b1_matrix(1, layer->feed_forward->b1.size());
        Matrix b2_matrix(1, layer->feed_forward->b2.size());
        for (size_t i = 0; i < layer->feed_forward->b1.size(); ++i) {
            b1_matrix(0, i) = layer->feed_forward->b1[i];
        }
        for (size_t i = 0; i < layer->feed_forward->b2.size(); ++i) {
            b2_matrix(0, i) = layer->feed_forward->b2[i];
        }
        all_params.push_back(b1_matrix);
        all_params.push_back(b2_matrix);
        
        // Add final layer norm parameters
        Matrix ffn_gamma_matrix(1, layer->ffn_ln->gamma.size());
        Matrix ffn_beta_matrix(1, layer->ffn_ln->beta.size());
        for (size_t i = 0; i < layer->ffn_ln->gamma.size(); ++i) {
            ffn_gamma_matrix(0, i) = layer->ffn_ln->gamma[i];
            ffn_beta_matrix(0, i) = layer->ffn_ln->beta[i];
        }
        all_params.push_back(ffn_gamma_matrix);
        all_params.push_back(ffn_beta_matrix);
    }
    
    // Add final layer norm parameters
    Matrix final_gamma_matrix(1, final_ln->gamma.size());
    Matrix final_beta_matrix(1, final_ln->beta.size());
    for (size_t i = 0; i < final_ln->gamma.size(); ++i) {
        final_gamma_matrix(0, i) = final_ln->gamma[i];
        final_beta_matrix(0, i) = final_ln->beta[i];
    }
    all_params.push_back(final_gamma_matrix);
    all_params.push_back(final_beta_matrix);
    
    return all_params;
}

void Transformer::save(std::ostream& os) const {
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

void Transformer::load(std::istream& is) {
    // Load config
    is.read(reinterpret_cast<char*>(&config), sizeof(config));
    
    // Load embeddings
    token_embedding = TokenEmbedding::load(is);
    pos_encoding = PositionalEncoding::load(is);
    
    // Load layers
    layers.clear();
    for (size_t i = 0; i < config.num_layers; ++i) {
        layers.push_back(TransformerLayer::load(is));
    }
    
    // Load final layer norm
    final_ln = LayerNorm::load(is);
    
#ifdef USE_CUDA
    if (config.use_cuda) {
        cuda_manager = std::make_unique<CudaManager>();
    }
#endif
}

Matrix TransformerLayer::backward(const Matrix& grad, const Matrix& input) const {
    // Backward through feed forward
    Matrix d_residual2 = grad;
    Matrix d_ffn = feed_forward->backward(d_residual2, ffn_ln->forward(input));
    Matrix d_ln2 = ffn_ln->backward(d_ffn, input);
    
    // Backward through attention
    Matrix d_residual1 = d_ln2 + d_residual2;
    Matrix d_attn = self_attention->backward(d_residual1, attention_ln->forward(input));
    Matrix d_ln1 = attention_ln->backward(d_attn, input);
    
    return d_ln1;
} 