#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "token_constants.hpp"
#include "base_tokenizer.hpp"

// Forward declaration
class TiktokenTokenizer;

class Tokenizer {
private:
    std::unique_ptr<BaseTokenizer> tokenizer_;

public:
    explicit Tokenizer(const std::string& encoding_name = "gpt2");
    ~Tokenizer() = default;

    // Core tokenization methods
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    size_t vocab_size() const;

    // Virtual methods with default implementations
    virtual void preprocess_text(std::string& text) const;
    virtual bool is_special_token(int token_id) const;

    // Special token accessors
    virtual int get_pad_token_id() const { return 0; }
    virtual int get_unk_token_id() const { return 1; }
    virtual int get_bos_token_id() const { return 2; }
    virtual int get_eos_token_id() const { return 3; }
    virtual int get_mask_token_id() const { return 4; }
    virtual int get_sep_token_id() const { return 5; }

    // Token type checking methods
    virtual bool is_verb(const std::string& token) const;
    virtual bool is_adjective(const std::string& token) const;
    virtual bool is_noun(const std::string& token) const;
    virtual bool is_determiner(const std::string& token) const;

    // Debug helpers
    virtual void print_vocabulary_mappings() const;

    // Serialization
    virtual void save(std::ostream& os) const;
    static std::unique_ptr<Tokenizer> load(std::istream& is);
    virtual std::vector<std::string> get_vocabulary_vector() const;

    void initialize(const std::string& encoding_name = "cl100k_base");

    // Add method to set vocabulary size
    void set_vocab_size(size_t size) {
        if (tokenizer_) {
            tokenizer_->set_vocab_size(size);
        }
    }

protected:
    // Token category sets
    std::unordered_set<std::string> verb_tokens_;
    std::unordered_set<std::string> adjective_tokens_;
    std::unordered_set<std::string> noun_tokens_;
    std::unordered_set<std::string> determiner_tokens_;

    // Special character mapping for preprocessing
    static const std::unordered_map<char, std::string> SPECIAL_CHAR_MAP;

private:
    // Add private helper methods
    void save_vocabulary(std::ostream& os) const;
    std::unique_ptr<TiktokenTokenizer> load_vocabulary(std::istream& is);
};