#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <sentencepiece_processor.h>

class Tokenizer {
private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> processor;
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<int, std::string> id_to_token;
    size_t vocab_size_;
    
public:
    explicit Tokenizer(const std::string& model_path);
    
    // Tokenization
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    
    // Vocabulary management
    size_t vocab_size() const { return vocab_size_; }
    void save(const std::string& path) const;
    static Tokenizer load(const std::string& path);
}; 