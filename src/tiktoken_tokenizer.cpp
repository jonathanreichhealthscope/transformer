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

TiktokenTokenizer::TiktokenTokenizer() = default;

void TiktokenTokenizer::initialize(const std::string& encoding_name) {
    try {
        // Initialize the base tiktoken encoder with GPT-2 encoding
        tiktoken_ = std::make_unique<tiktoken::Encoding>("gpt2");
        
        // Get the executable's directory
        std::filesystem::path exe_path = std::filesystem::current_path();
        
        // Try to find training data file using same path resolution as vocab file
        std::ifstream train_file;
        std::string train_path;
        
        // First try data directory relative to executable
        auto training_data_path = exe_path / "data" / "training_pairs.txt";
        train_file.open(training_data_path);
        if (train_file.is_open()) {
            train_path = training_data_path.string();
            std::cout << "Using training data from: " << train_path << std::endl;
        } else {
            // Try one directory up (in case running from build directory)
            train_file.close();
            train_file.clear();
            auto parent_training_path = exe_path.parent_path() / "data" / "training_pairs.txt";
            train_file.open(parent_training_path);
            if (train_file.is_open()) {
                train_path = parent_training_path.string();
                std::cout << "Using training data from: " << train_path << std::endl;
            } else {
                throw std::runtime_error("Could not find training_pairs.txt in data directory");
            }
        }
        
        // First, collect token frequency statistics from the training data
        std::unordered_map<int, size_t> token_frequencies;
        std::unordered_map<std::string, size_t> word_frequencies;
        
        std::string line;
        size_t total_tokens = 0;
        while (std::getline(train_file, line)) {
            if (line.empty()) continue;
            
            // Count raw words
            std::istringstream iss(line);
            std::string word;
            while (iss >> word) {
                word_frequencies[word]++;
            }
            
            // Count BPE tokens
            auto tokens = tiktoken_->encode(line);
            total_tokens += tokens.size();
            for (int token : tokens) {
                token_frequencies[token]++;
            }
        }
        
        std::cout << "\nToken Analysis:" << std::endl;
        std::cout << "Raw word count: " << word_frequencies.size() << " unique words" << std::endl;
        std::cout << "BPE token count: " << token_frequencies.size() << " unique tokens" << std::endl;
        std::cout << "Analyzed " << total_tokens << " total BPE tokens in training data" << std::endl;
        
        // Initialize ID mappings
        // First add special tokens (they keep their original IDs 0-4)
        for (int i = 0; i < 5; i++) {
            old_to_new_id_[i] = i;
            new_to_old_id_[i] = i;
        }
        
        // For all other tokens in the training data, map them to consecutive IDs
        size_t next_id = 5;  // Start after special tokens
        for (const auto& [token_id, freq] : token_frequencies) {
            if (token_id >= 5) {  // Skip special tokens
                old_to_new_id_[token_id] = next_id;
                new_to_old_id_[next_id] = token_id;
                next_id++;
            }
        }
        
        target_vocab_size = next_id;  // Set the actual vocabulary size
        
        std::cout << "\nVocabulary Statistics:" << std::endl;
        std::cout << "Using " << next_id << " tokens from training data" << std::endl;
        std::cout << "Token ID range: [0, " << next_id << ")" << std::endl;
        
        // Print frequency distribution and example tokens
        if (!token_frequencies.empty()) {
            std::cout << "\nTop 10 most frequent BPE tokens:" << std::endl;
            std::vector<std::pair<int, size_t>> freq_vec(token_frequencies.begin(), token_frequencies.end());
            std::sort(freq_vec.begin(), freq_vec.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
            
            for (size_t i = 0; i < std::min(size_t(10), freq_vec.size()); i++) {
                std::string token_text = tiktoken_->decode({freq_vec[i].first});
                std::cout << "  " << freq_vec[i].first << " -> " << next_id 
                         << " ('" << token_text << "'): " 
                         << freq_vec[i].second << " occurrences" << std::endl;
            }
            
            // Show some example word tokenizations
            std::cout << "\nExample word tokenizations:" << std::endl;
            size_t examples = 0;
            for (const auto& [word, _] : word_frequencies) {
                if (examples >= 5) break;
                auto word_tokens = tiktoken_->encode(word);
                std::cout << "'" << word << "' -> ";
                for (size_t i = 0; i < word_tokens.size(); i++) {
                    std::string token_text = tiktoken_->decode({word_tokens[i]});
                    std::cout << "'" << token_text << "'(" << word_tokens[i] << ")";
                    if (i < word_tokens.size() - 1) std::cout << " + ";
                }
                std::cout << std::endl;
                examples++;
            }
        }
        
        // Setup special tokens
        setup_special_tokens();
        
        std::cout << "Final vocabulary breakdown:" << std::endl;
        std::cout << "- Special tokens: 5" << std::endl;
        std::cout << "- Regular tokens: " << (token_frequencies.size() - 5) << std::endl;
        std::cout << "Total vocabulary size: " << vocab_size() << " tokens" << std::endl;
        
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
        if (text.empty()) {
            if (debug_logging_) {
                std::cout << "Warning: Attempting to encode empty string" << std::endl;
            }
            return std::vector<int>();
        }

        // Use tiktoken's encode method to get original token IDs
        auto old_tokens = tiktoken_->encode(text);
        
        // Debug logging - only if enabled
        if (debug_logging_) {
            std::cout << "Encoding text: '" << text << "'" << std::endl;
            std::cout << "Original tokens: ";
            // Cache decoded tokens to avoid repeated decoding
            std::unordered_map<int, std::string> decoded_cache;
            for (int t : old_tokens) {
                if (decoded_cache.find(t) == decoded_cache.end()) {
                    decoded_cache[t] = tiktoken_->decode({t});
                }
                std::cout << t << "(" << decoded_cache[t] << ") ";
            }
            std::cout << std::endl;
        }
        
        // Convert to our new token IDs
        std::vector<int> new_tokens;
        new_tokens.reserve(old_tokens.size() + 2);
        new_tokens.push_back(tokens::BOS_ID);
        
        size_t unk_count = 0;
        for (int old_id : old_tokens) {
            int new_id = convert_to_new_id(old_id);
            if (new_id == tokens::UNK_ID) {
                unk_count++;
                if (debug_logging_) {
                    std::cout << "Token " << old_id << " mapped to UNK" << std::endl;
                }
            }
            new_tokens.push_back(new_id);
        }
        
        new_tokens.push_back(tokens::EOS_ID);
        
        if (debug_logging_ && unk_count > 0) {
            std::cout << "Replaced " << unk_count << " tokens with <unk> token" << std::endl;
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
        // Convert our new token IDs back to original IDs
        std::vector<int> old_tokens;
        old_tokens.reserve(new_tokens.size());
        
        for (int new_id : new_tokens) {
            if (new_id != tokens::BOS_ID && new_id != tokens::EOS_ID) {
                old_tokens.push_back(convert_to_old_id(new_id));
            }
        }
        
        // Use tiktoken's decode method with original IDs
        return tiktoken_->decode(old_tokens);
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