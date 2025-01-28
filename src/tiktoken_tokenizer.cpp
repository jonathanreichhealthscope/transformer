#include "../include/tiktoken_tokenizer.hpp"
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <random>
#include <algorithm>
#include <iomanip>

TiktokenTokenizer::TiktokenTokenizer() = default;

void TiktokenTokenizer::initialize(const std::string& encoding_name) {
    try {
        std::cout << "Initializing tokenizer..." << std::endl;
        
        // Initialize with gpt2 encoding
        tiktoken_ = std::make_unique<tiktoken::Encoding>("gpt2");
        std::cout << "Loaded gpt2 vocabulary with " << tiktoken_->get_vocab_size() << " unique tokens" << std::endl;
        
        // First, collect all GPT2 tokens and their IDs
        std::vector<std::pair<std::string, int>> gpt2_tokens;
        std::string sample_text = "This is a sample text to help build the vocabulary.";
        std::vector<int> sample_ids = tiktoken_->encode(sample_text);
        
        // Create a set of unique token IDs
        std::unordered_set<int> seen_ids;
        
        // Process sample text tokens
        for (const auto& id : sample_ids) {
            if (seen_ids.insert(id).second) {  // If this is a new ID
                std::string token = tiktoken_->decode({id});
                gpt2_tokens.push_back({token, id});
            }
        }
        
        // Add more tokens by sampling individual characters and common combinations
        std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-_'\"();:$/ ";
        for (char c : chars) {
            std::string token_str(1, c);
            std::vector<int> ids = tiktoken_->encode(token_str);
            for (const auto& id : ids) {
                if (seen_ids.insert(id).second) {
                    std::string token = tiktoken_->decode({id});
                    gpt2_tokens.push_back({token, id});
                }
            }
        }
        
        // Sort tokens by their GPT2 IDs
        std::sort(gpt2_tokens.begin(), gpt2_tokens.end(),
                 [](const auto& a, const auto& b) { return a.second < b.second; });
        
        // Create our token mappings
        old_to_new_id_.clear();
        new_to_old_id_.clear();
        
        // First, map special tokens
        setup_special_tokens();
        for (int i = 0; i < 5; i++) {
            old_to_new_id_[i] = i;
            new_to_old_id_[i] = i;
        }
        
        // Then map GPT2 tokens to our consecutive IDs
        int current_id = 5;  // Start after special tokens
        for (const auto& [token, gpt2_id] : gpt2_tokens) {
            if (current_id >= target_vocab_size) break;
            
            // Skip if this is a special token
            if (gpt2_id < 5) continue;
            
            old_to_new_id_[gpt2_id] = current_id;
            new_to_old_id_[current_id] = gpt2_id;
            current_id++;
        }
        
        std::cout << "Vocabulary mapping complete:" << std::endl;
        std::cout << "- Total mapped tokens: " << current_id << std::endl;
        std::cout << "- Special tokens: 5" << std::endl;
        std::cout << "- Regular GPT-2 tokens: " << (current_id - 5) << std::endl;
        std::cout << "- Total GPT-2 vocabulary size: " << tiktoken_->get_vocab_size() << std::endl;
        
        // Initialize token frequencies
        token_frequencies_.clear();
        for (const auto& [token, gpt2_id] : gpt2_tokens) {
            if (old_to_new_id_.find(gpt2_id) != old_to_new_id_.end()) {
                // More frequent tokens appear earlier in GPT2's vocabulary
                float freq = 1.0f - (static_cast<float>(gpt2_id) / tiktoken_->get_vocab_size());
                token_frequencies_[token] = freq;
            }
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize tokenizer: " + std::string(e.what()));
    }
}

void TiktokenTokenizer::setup_special_tokens() {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }

    // Add our special tokens to tiktoken's vocabulary in the same order as defined in token_constants.hpp
    tiktoken_->add_special_token("<pad>", tokens::PAD_ID);    // ID 0
    tiktoken_->add_special_token("<unk>", tokens::UNK_ID);    // ID 1
    tiktoken_->add_special_token("<s>", tokens::BOS_ID);      // ID 2
    tiktoken_->add_special_token("</s>", tokens::EOS_ID);     // ID 3
    tiktoken_->add_special_token("<mask>", tokens::MASK_ID);  // ID 4
}

int TiktokenTokenizer::convert_to_new_id(int old_id) const {
    if (old_id < 5) return old_id;  // Special tokens keep their IDs
    auto it = old_to_new_id_.find(old_id);
    return it != old_to_new_id_.end() ? it->second : tokens::UNK_ID;
}

int TiktokenTokenizer::convert_to_old_id(int new_id) const {
    if (new_id < 5) return new_id;  // Special tokens keep their IDs
    auto it = new_to_old_id_.find(new_id);
    return it != new_to_old_id_.end() ? it->second : tokens::UNK_ID;
}

// Add debug flag as a static member
bool TiktokenTokenizer::debug_logging_ = false;

// Add static method to control debug logging
void TiktokenTokenizer::set_debug_logging(bool enable) {
    debug_logging_ = enable;
}

std::vector<int> TiktokenTokenizer::encode(const std::string& text) const {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }

    try {
        // Get original token IDs from tiktoken
        std::vector<int> old_tokens = tiktoken_->encode(text);
        std::vector<int> new_tokens;
        new_tokens.reserve(old_tokens.size());
        
        // Convert to our new token IDs
        for (int old_id : old_tokens) {
            new_tokens.push_back(convert_to_new_id(old_id));
        }
        
        return new_tokens;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to encode text: " + std::string(e.what()));
    }
}

std::string TiktokenTokenizer::decode(const std::vector<int>& new_tokens) const {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }

    try {
        std::string result;
        bool first_token = true;
        
        for (int new_id : new_tokens) {
            int old_id = convert_to_old_id(new_id);
            std::string token = tiktoken_->decode({old_id});
            
            // Handle the token
            if (!token.empty()) {
                if (token[0] == 'Ġ') {
                    // Add space only if it's not the first token
                    if (!first_token) {
                        result += ' ';
                    }
                    result += token.substr(1);  // Add rest of token without Ġ
                } else {
                    result += token;
                }
            }
            first_token = false;
        }
        
        return result;
    } catch (const std::exception& e) {
        throw std::runtime_error("Decoding failed: " + std::string(e.what()));
    }
}

size_t TiktokenTokenizer::vocab_size() const {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    // Return the actual filtered vocabulary size (target size + special tokens)
    return target_vocab_size;
} 