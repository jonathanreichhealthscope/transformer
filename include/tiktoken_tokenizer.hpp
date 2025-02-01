#pragma once
#include "base_tokenizer.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "tiktoken/tiktoken/tiktoken.hpp"
#include "token_constants.hpp"

class TiktokenTokenizer : public BaseTokenizer {
public:
    explicit TiktokenTokenizer(const std::string& encoding_name = "gpt2");
    ~TiktokenTokenizer() override;

    // Core tokenization methods from BaseTokenizer
    std::vector<int> encode(const std::string& text) const override;
    std::string decode(const std::vector<int>& tokens) const override;
    size_t vocab_size() const override;
    void initialize(const std::string& encoding_name) override;
    bool is_initialized() const override;

    // Special token getters from BaseTokenizer
    int get_pad_token_id() const override { return tokens::PAD_ID; }
    int get_unk_token_id() const override { return tokens::UNK_ID; }
    int get_bos_token_id() const override { return tokens::BOS_ID; }
    int get_eos_token_id() const override { return tokens::EOS_ID; }
    int get_mask_token_id() const override { return tokens::MASK_ID; }

    // Additional methods specific to TiktokenTokenizer
    std::vector<int> encode(const std::string& text, bool add_special_tokens) const;
    std::string decode(const std::vector<int>& tokens, bool skip_special_tokens) const;
    void print_vocabulary_mappings() const;
    void save(std::ostream& os) const;
    std::vector<std::string> get_vocabulary_vector() const;
    int get_sep_token_id() const { return tokens::SEP_ID; }

    // Vocabulary management
    void set_vocab_size(size_t size) override;
    void print_vocabulary_stats() const;  // New method for debugging

    // Configuration
    static void set_debug_logging(bool enable) { debug_logging_ = enable; }
    static constexpr const char* SEP_TOKEN = "|";

protected:
    // Token categories inherited from Tokenizer
    std::unordered_set<std::string> verb_tokens_;
    std::unordered_set<std::string> adjective_tokens_;
    std::unordered_set<std::string> noun_tokens_;
    std::unordered_set<std::string> determiner_tokens_;
    void initialize_token_categories();

private:
    // Forward declare private implementation
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
    bool initialized_ = false;
    std::vector<bool> filtered_tokens_;
    std::unordered_map<int, int> old_to_new_id_;
    std::unordered_map<int, int> new_to_old_id_;
    std::unordered_map<std::string, size_t> token_frequencies_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;
    size_t target_vocab_size = 2500;
    static bool debug_logging_;
    size_t vocab_size_;

    // Private helpers
    void setup_special_tokens();
    int convert_to_new_id(int old_id) const;
    int convert_to_old_id(int new_id) const;
    void build_vocabulary_from_frequencies();
    std::vector<int> tokenize_text(const std::string& text) const;
    std::string decode_token(int token_id) const;
}; 