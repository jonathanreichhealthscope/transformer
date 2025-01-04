#pragma once
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "vocabulary.hpp"

class Tokenizer {
public:
    Tokenizer();
    
    // Core tokenization functions
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    void preprocess_text(std::string& text) const;
    bool is_special_token(int token_id) const;
    
    // Existing functionality
    void save(std::ostream& os) const;
    static std::unique_ptr<Tokenizer> load(std::istream& is);
    size_t vocab_size() const { return vocab->size(); }
    void print_vocabulary_mappings() const { vocab->print_vocabulary_mappings(); }
    bool verify_mappings() const { return vocab->verify_mappings(); }
    
    // Add has_token method
    bool has_token(const std::string& token) const { return vocab->has_token(token); }
    
    void clear_cache() { encoding_cache.clear(); }
    
    // Add getter for pad token
    int get_pad_token_id() const { return vocab->get_pad_token_id(); }
    
    const Vocabulary& get_vocabulary() const { return *vocab; }
    
private:
    std::unique_ptr<Vocabulary> vocab;
    mutable std::unordered_map<std::string, std::vector<int>> encoding_cache;
    static constexpr size_t MAX_SUBWORD_LENGTH = 32;
    
    void save_vocabulary(std::ostream& os) const;
    static std::unique_ptr<Vocabulary> load_vocabulary(std::istream& is);
};

extern const std::unordered_map<char, std::string> SPECIAL_CHAR_MAP;