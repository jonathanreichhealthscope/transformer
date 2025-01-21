#pragma once
#include "vocabulary.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

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
    size_t vocab_size() const {
        return vocab->size();
    }
    void print_vocabulary_mappings() const {
        vocab->print_vocabulary_mappings();
    }
    bool verify_mappings() const {
        return vocab->verify_mappings();
    }

    // Add has_token method
    bool has_token(const std::string& token) const {
        return vocab->has_token(token);
    }

    void clear_cache() {
        encoding_cache.clear();
    }

    // Special token getters - complete set
    int get_pad_token_id() const {
        return vocab->get_pad_token_id();
    }
    int get_unk_token_id() const {
        return vocab->get_unk_token_id();
    }
    int get_bos_token_id() const {
        return vocab->get_bos_token_id();
    }
    int get_eos_token_id() const {
        return vocab->get_eos_token_id();
    }
    int get_mask_token_id() const {
        return vocab->get_mask_token_id();
    }

    const Vocabulary& get_vocabulary() const {
        return *vocab;
    }

    // Add a method to access the map if needed
    static const std::unordered_map<char, std::string>& get_special_char_map() {
        return SPECIAL_CHAR_MAP;
    }

  private:
    std::unique_ptr<Vocabulary> vocab;
    mutable std::unordered_map<std::string, std::vector<int>> encoding_cache;
    static constexpr size_t MAX_SUBWORD_LENGTH = 32;

    void save_vocabulary(std::ostream& os) const;
    static std::unique_ptr<Vocabulary> load_vocabulary(std::istream& is);

    // Add the special character map as a static const member
    static const std::unordered_map<char, std::string> SPECIAL_CHAR_MAP;
};