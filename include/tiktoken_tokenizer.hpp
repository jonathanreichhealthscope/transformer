#pragma once
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
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

    void set_vocab_size(size_t size) {
        target_vocab_size = size;
    }

    static void set_debug_logging(bool enable);  // Add static method to control logging

private:
    std::unique_ptr<tiktoken::Encoding> tiktoken_;
    std::vector<bool> filtered_tokens_;  // Tracks which tokens we keep
    std::unordered_map<int, int> old_to_new_id_;  // Maps original token IDs to our new consecutive IDs
    std::unordered_map<int, int> new_to_old_id_;  // Maps our new consecutive IDs back to original token IDs
    std::unordered_map<std::string, size_t> token_frequencies_;  // Track token frequencies for vocabulary selection
    
    // Map between our special token IDs and tiktoken's vocabulary
    void setup_special_tokens();
    
    // Helper to convert between old and new token IDs
    int convert_to_new_id(int old_id) const;
    int convert_to_old_id(int new_id) const;

    // Helper to build frequency-based vocabulary
    void build_vocabulary_from_frequencies();

    size_t target_vocab_size = 2500;  // Reduced from 7000 to better match training data distribution
    static bool debug_logging_;  // Add static debug flag
}; 