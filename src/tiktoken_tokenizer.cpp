#include "../include/tiktoken_tokenizer.hpp"
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <unordered_map>
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
        std::cout << "Loaded gpt2 vocabulary" << std::endl;
        
        // Set target vocabulary size
        target_vocab_size = 2500;
        
        // Initialize token frequencies with cleaned tokens
        std::vector<std::string> common_words = {
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me"
        };
        
        // Add common word combinations
        std::vector<std::string> word_combinations = {
            "I am", "I will", "I have", "the cat", "to the", "in the", "of the",
            "it is", "they are", "we are", "you are", "he is", "she is",
            "school", "home", "work", "store", "market", "park", "library",
            "morning", "afternoon", "evening", "night"
        };
        
        common_words.insert(common_words.end(), word_combinations.begin(), word_combinations.end());
        
        // First add special tokens
        setup_special_tokens();
        
        // Initialize our token mappings
        std::cout << "Building vocabulary mappings..." << std::endl;
        int current_id = 5;  // Start after special tokens
        
        // First, add special tokens to our mappings
        for (int i = 0; i < 5; i++) {
            old_to_new_id_[i] = i;
            new_to_old_id_[i] = i;
        }
        
        std::cout << "Adding common words to vocabulary..." << std::endl;
        
        // Process common words first
        for (const auto& word : common_words) {
            // Add space prefix for encoding if not present
            std::string search_word = (word[0] != ' ') ? " " + word : word;
            std::vector<int> token_ids = tiktoken_->encode(search_word);
            
            // Only add single-token words to maintain vocabulary efficiency
            if (token_ids.size() == 1 && current_id < target_vocab_size) {
                int old_id = token_ids[0];
                if (old_to_new_id_.find(old_id) == old_to_new_id_.end()) {
                    old_to_new_id_[old_id] = current_id;
                    new_to_old_id_[current_id] = old_id;
                    current_id++;
                }
            }
        }
        
        std::cout << "Adding remaining tokens..." << std::endl;
        
        // Add remaining common tokens by sampling text
        std::string sample_text = "This is a sample text to help build the vocabulary. "
                                "It includes common words and patterns that might be useful. "
                                "We want to ensure we have good coverage of typical English text. "
                                "The quick brown fox jumps over the lazy dog. "
                                "Numbers like 0 1 2 3 4 5 6 7 8 9 are important too. "
                                "Common punctuation , . ! ? ( ) [ ] { } : ; ' \" should be included.";
        
        std::vector<int> sample_tokens = tiktoken_->encode(sample_text);
        for (int old_id : sample_tokens) {
            if (current_id >= target_vocab_size) break;
            if (old_to_new_id_.find(old_id) == old_to_new_id_.end()) {
                old_to_new_id_[old_id] = current_id;
                new_to_old_id_[current_id] = old_id;
                current_id++;
            }
        }
        
        std::cout << "Vocabulary construction complete:" << std::endl;
        std::cout << "- Total tokens: " << current_id << std::endl;
        std::cout << "- Special tokens: 5" << std::endl;
        std::cout << "- Regular tokens: " << (current_id - 5) << std::endl;
        
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