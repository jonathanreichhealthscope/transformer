#include "../include/tokenizer.hpp"
#include <fstream>
#include <stdexcept>

Tokenizer::Tokenizer(const std::string& model_path) {
    // Initialize SentencePiece processor
    processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
    const auto status = processor->Load(model_path);
    
    if (!status.ok()) {
        throw std::runtime_error("Failed to load SentencePiece model: " + status.ToString());
    }
    
    // Initialize vocabulary size
    vocab_size_ = processor->GetPieceSize();
    
    // Build token mappings
    for (int i = 0; i < vocab_size_; ++i) {
        const auto& piece = processor->IdToPiece(i);
        token_to_id[piece] = i;
        id_to_token[i] = piece;
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    if (!processor->Encode(text, &ids).ok()) {
        throw std::runtime_error("Failed to encode text");
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string text;
    if (!processor->Decode(tokens, &text).ok()) {
        throw std::runtime_error("Failed to decode tokens");
    }
    return text;
}

void Tokenizer::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for saving tokenizer");
    }
    
    // Save vocab size
    file.write(reinterpret_cast<const char*>(&vocab_size_), sizeof(vocab_size_));
    
    // Save token mappings
    size_t map_size = token_to_id.size();
    file.write(reinterpret_cast<const char*>(&map_size), sizeof(map_size));
    
    for (const auto& [token, id] : token_to_id) {
        size_t token_length = token.length();
        file.write(reinterpret_cast<const char*>(&token_length), sizeof(token_length));
        file.write(token.c_str(), token_length);
        file.write(reinterpret_cast<const char*>(&id), sizeof(id));
    }
}

Tokenizer Tokenizer::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for loading tokenizer");
    }
    
    // Create tokenizer with empty model path
    Tokenizer tokenizer("");
    
    // Load vocab size
    file.read(reinterpret_cast<char*>(&tokenizer.vocab_size_), sizeof(tokenizer.vocab_size_));
    
    // Load token mappings
    size_t map_size;
    file.read(reinterpret_cast<char*>(&map_size), sizeof(map_size));
    
    for (size_t i = 0; i < map_size; ++i) {
        size_t token_length;
        file.read(reinterpret_cast<char*>(&token_length), sizeof(token_length));
        
        std::string token(token_length, '\0');
        file.read(&token[0], token_length);
        
        int id;
        file.read(reinterpret_cast<char*>(&id), sizeof(id));
        
        tokenizer.token_to_id[token] = id;
        tokenizer.id_to_token[id] = token;
    }
    
    return tokenizer;
} 