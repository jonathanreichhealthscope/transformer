#include "../include/transformer.hpp"
#include <fstream>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>

void Transformer::save_model(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    cereal::BinaryOutputArchive archive(os);
    
    // Save config
    archive(config);
    
    // Save layer weights
    for (const auto& layer : layers) {
        layer->save(archive);
    }
    
    // Save embeddings
    token_embedding->save(archive);
    pos_encoding->save(archive);
    
    // Save final layer norm
    final_ln->save(archive);
}

Transformer Transformer::load_model(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    cereal::BinaryInputArchive archive(is);
    
    // Load config
    TransformerConfig config;
    archive(config);
    
    // Create model
    Transformer model(config);
    
    // Load layer weights
    for (auto& layer : model.layers) {
        layer->load(archive);
    }
    
    // Load embeddings
    model.token_embedding->load(archive);
    model.pos_encoding->load(archive);
    
    // Load final layer norm
    model.final_ln->load(archive);
    
    return model;
} 