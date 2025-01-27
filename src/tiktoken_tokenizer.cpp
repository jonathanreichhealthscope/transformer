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
        // Initialize the base tiktoken encoder
        tiktoken_ = std::make_unique<tiktoken::Encoding>("gpt2");
        
        // Get the executable's directory
        std::filesystem::path exe_path = std::filesystem::current_path();
        
        // Try both possible locations for the vocabulary file
        std::ifstream vocab_file;
        std::string vocab_path;
        
        // First try data directory relative to executable
        auto data_path = exe_path / "data" / "tiktoken_data" / "gpt2.vocab.json";
        vocab_file.open(data_path);
        if (vocab_file.is_open()) {
            vocab_path = data_path.string();
            std::cout << "Using GPT-2 vocabulary from: " << vocab_path << std::endl;
        } else {
            // Try build directory if data directory failed
            vocab_file.close();
            vocab_file.clear();
            auto build_path = exe_path / "build" / "tiktoken_data" / "gpt2.vocab.json";
            vocab_file.open(build_path);
            if (vocab_file.is_open()) {
                vocab_path = build_path.string();
                std::cout << "Using GPT-2 vocabulary from: " << vocab_path << std::endl;
            } else {
                // Try one directory up (in case running from build directory)
                vocab_file.close();
                vocab_file.clear();
                auto parent_data_path = exe_path.parent_path() / "data" / "tiktoken_data" / "gpt2.vocab.json";
                vocab_file.open(parent_data_path);
                if (vocab_file.is_open()) {
                    vocab_path = parent_data_path.string();
                    std::cout << "Using GPT-2 vocabulary from: " << vocab_path << std::endl;
                } else {
                    throw std::runtime_error("Could not open GPT-2 vocabulary file in either data/tiktoken_data/ or build/tiktoken_data/");
                }
            }
        }
        
        // Load and parse vocabulary
        nlohmann::json vocab_json;
        vocab_file >> vocab_json;
        
        // Convert vocabulary to our format
        std::unordered_map<std::string, int> vocab;
        for (const auto& [token, id] : vocab_json.items()) {
            vocab[token] = id.get<int>();
        }
        
        std::cout << "Loaded GPT-2 vocabulary with " << vocab.size() << " tokens" << std::endl;
        
        // First, collect token frequency statistics from the training data
        std::unordered_map<int, size_t> token_frequencies;
        
        // Read training data file
        std::ifstream train_file("data/train.txt");
        if (!train_file.is_open()) {
            throw std::runtime_error("Could not open training data file");
        }
        
        std::string line;
        size_t total_tokens = 0;
        while (std::getline(train_file, line)) {
            if (line.empty()) continue;
            auto tokens = tiktoken_->encode(line);
            total_tokens += tokens.size();
            for (int token : tokens) {
                token_frequencies[token]++;
            }
        }
        
        std::cout << "Analyzed " << total_tokens << " total tokens in training data" << std::endl;
        std::cout << "Found " << token_frequencies.size() << " unique tokens in training data" << std::endl;
        
        // Convert frequencies to vector for sorting
        std::vector<std::pair<int, size_t>> token_freq_vec;
        for (const auto& [token, freq] : token_frequencies) {
            if (token >= 5) {  // Skip special tokens
                token_freq_vec.push_back({token, freq});
            }
        }
        
        // Sort by frequency, highest first
        std::sort(token_freq_vec.begin(), token_freq_vec.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Take most frequent tokens up to our target size
        size_t filtered_vocab_size = 5;  // Start after special tokens
        
        // First add special tokens (they keep their original IDs 0-4)
        for (int i = 0; i < 5; i++) {
            old_to_new_id_[i] = i;
            new_to_old_id_[i] = i;
        }
        
        // Calculate minimum frequency threshold (tokens that appear at least 5 times)
        const size_t MIN_FREQUENCY = 5;
        size_t tokens_above_threshold = 0;
        for (const auto& [token, freq] : token_freq_vec) {
            if (freq >= MIN_FREQUENCY) tokens_above_threshold++;
        }
        
        std::cout << "Tokens appearing >= " << MIN_FREQUENCY << " times: " << tokens_above_threshold << std::endl;
        
        // Add frequent tokens
        for (const auto& [old_id, freq] : token_freq_vec) {
            if (freq >= MIN_FREQUENCY) {
                old_to_new_id_[old_id] = filtered_vocab_size;
                new_to_old_id_[filtered_vocab_size] = old_id;
                filtered_vocab_size++;
            }
        }
        
        target_vocab_size = filtered_vocab_size;
        
        std::cout << "Using " << filtered_vocab_size << " tokens after filtering" << std::endl;
        std::cout << "Token frequency distribution:" << std::endl;
        
        // Print frequency distribution
        if (!token_freq_vec.empty()) {
            std::cout << "Top 10 most frequent tokens:" << std::endl;
            for (size_t i = 0; i < std::min(size_t(10), token_freq_vec.size()); i++) {
                std::string token_text = tiktoken_->decode({token_freq_vec[i].first});
                std::cout << "  " << token_text << ": " << token_freq_vec[i].second << " occurrences" << std::endl;
            }
        }
        
        // Setup special tokens and print final stats
        setup_special_tokens();
        
        std::cout << "Final vocabulary breakdown:" << std::endl;
        std::cout << "- Special tokens: 5" << std::endl;
        std::cout << "- Regular tokens: " << (filtered_vocab_size - 5) << std::endl;
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