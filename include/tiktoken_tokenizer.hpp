#pragma once
#include <string>
#include <vector>
#include <memory>
#include "tiktoken/tiktoken/tiktoken.hpp"
#include "token_constants.hpp"

class TiktokenTokenizer {
public:
    TiktokenTokenizer();
    ~TiktokenTokenizer() = default;

    // Core tokenization methods
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;

    // Special token getters - using our constants
    int get_pad_token_id() const { return tokens::PAD_ID; }
    int get_unk_token_id() const { return tokens::UNK_ID; }
    int get_bos_token_id() const { return tokens::BOS_ID; }
    int get_eos_token_id() const { return tokens::EOS_ID; }
    int get_mask_token_id() const { return tokens::MASK_ID; }

    // Vocabulary size
    size_t vocab_size() const;

    // Initialize with model type
    void initialize(const std::string& encoding_name = "cl100k_base");

    bool is_initialized() const { return tiktoken_ != nullptr; }

private:
    std::unique_ptr<tiktoken::Encoding> tiktoken_;
    
    // Map between our special token IDs and tiktoken's vocabulary
    void setup_special_tokens();
}; 