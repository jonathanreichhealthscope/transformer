#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "tiktoken_tokenizer.hpp"

class Tokenizer {
public:
    // Default constructor
    Tokenizer() : encoding_name_("gpt2") {}
    
    // Constructor with encoding type
    explicit Tokenizer(const std::string& encoding) : encoding_name_(encoding) {}
    
    ~Tokenizer() = default;

    // Core tokenization methods
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    void preprocess_text(std::string& text) const;
    bool is_special_token(int token_id) const;

    // Special token accessors
    int get_pad_token_id() const { return tokenizer_->get_pad_token_id(); }
    int get_unk_token_id() const { return tokenizer_->get_unk_token_id(); }
    int get_bos_token_id() const { return tokenizer_->get_bos_token_id(); }
    int get_eos_token_id() const { return tokenizer_->get_eos_token_id(); }
    int get_mask_token_id() const { return tokenizer_->get_mask_token_id(); }
    
    size_t vocab_size() const { return tokenizer_->vocab_size(); }

    // Initialize tokenizer
    void initialize();

    bool is_initialized() const { return tokenizer_ && tokenizer_->is_initialized(); }

    // Debug helpers
    void print_vocabulary_mappings() const;

    /**
     * @brief Checks if a token is a noun
     * @param token Token string to check
     * @return True if the token is a noun
     */
    bool is_noun(const std::string& token) const;

    /**
     * @brief Checks if a token is an adjective
     * @param token Token string to check
     * @return True if the token is an adjective
     */
    bool is_adjective(const std::string& token) const;

    /**
     * @brief Checks if a token is a determiner
     * @param token Token string to check
     * @return True if the token is a determiner
     */
    bool is_determiner(const std::string& token) const;

    // Special character mapping for preprocessing
    static const std::unordered_map<char, std::string> SPECIAL_CHAR_MAP;

    // Add these declarations
    void save(std::ostream& os) const;
    static std::unique_ptr<Tokenizer> load(std::istream& is);
    std::vector<std::string> get_vocabulary_vector() const;

private:
    std::unique_ptr<TiktokenTokenizer> tokenizer_;
    std::string encoding_name_;
    
    // Vocabulary loading/saving helpers
    static std::unique_ptr<TiktokenTokenizer> load_vocabulary(std::istream& is);
    void save_vocabulary(std::ostream& os) const;
};