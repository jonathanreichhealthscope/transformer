#include "../include/tiktoken_tokenizer.hpp"
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>

TiktokenTokenizer::TiktokenTokenizer() = default;

void TiktokenTokenizer::initialize(const std::string& encoding_name) {
    try {
        // Initialize the base tiktoken encoder
        tiktoken_ = std::make_unique<tiktoken::Encoding>(encoding_name);
        
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
        } else {
            // Try build directory if data directory failed
            vocab_file.close();
            vocab_file.clear();
            auto build_path = exe_path / "build" / "tiktoken_data" / "gpt2.vocab.json";
            vocab_file.open(build_path);
            if (vocab_file.is_open()) {
                vocab_path = build_path.string();
            } else {
                // Try one directory up (in case running from build directory)
                vocab_file.close();
                vocab_file.clear();
                auto parent_data_path = exe_path.parent_path() / "data" / "tiktoken_data" / "gpt2.vocab.json";
                vocab_file.open(parent_data_path);
                if (vocab_file.is_open()) {
                    vocab_path = parent_data_path.string();
                } else {
                    throw std::runtime_error("Could not open GPT-2 vocabulary file in either data/tiktoken_data/ or build/tiktoken_data/");
                }
            }
        }
        
        std::cout << "Using vocabulary file at: " << vocab_path << std::endl;
        
        nlohmann::json vocab_json;
        vocab_file >> vocab_json;
        
        // Convert the vocabulary into pairs of (token, id)
        std::vector<std::pair<std::string, int>> vocab_pairs;
        for (const auto& [token, id] : vocab_json.items()) {
            vocab_pairs.emplace_back(token, id);
        }
        
        std::cout << "Loaded " << vocab_pairs.size() << " tokens from vocabulary file" << std::endl;
        
        // Sort by ID (lower IDs are more frequent in GPT-2)
        std::sort(vocab_pairs.begin(), vocab_pairs.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        
        // Initialize our filtered vocabulary
        filtered_tokens_.clear();
        filtered_tokens_.resize(vocab_pairs.size(), false);
        old_to_new_id_.clear();
        new_to_old_id_.clear();
        
        // Keep only the most frequent tokens up to target_vocab_size
        const size_t target_vocab_size = 3000;  // Reduced from 7000 to be closer to actual vocab size
        if (vocab_pairs.size() > target_vocab_size) {
            // Mark which tokens we're keeping in filtered_tokens_
            for (size_t i = 0; i < target_vocab_size; i++) {
                filtered_tokens_[vocab_pairs[i].second] = true;
            }
            
            // Create new ID mappings
            int new_id = 5;  // Start after special tokens
            for (size_t i = 0; i < target_vocab_size; i++) {
                int old_id = vocab_pairs[i].second;
                if (old_id >= 5) {  // Skip special token IDs
                    old_to_new_id_[old_id] = new_id;
                    new_to_old_id_[new_id] = old_id;
                    new_id++;
                }
            }
        }
        
        // Setup special tokens mapping
        setup_special_tokens();
        
        std::cout << "Filtered vocabulary size: " << vocab_size() << " tokens" << std::endl;
        
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

std::vector<int> TiktokenTokenizer::encode(const std::string& text) const {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }

    try {
        if (text.empty()) {
            std::cout << "Warning: Attempting to encode empty string" << std::endl;
            return std::vector<int>();
        }

        // Use tiktoken's encode method to get original token IDs
        auto old_tokens = tiktoken_->encode(text);
        
        // Debug logging
        std::cout << "Encoding text: '" << text << "'" << std::endl;
        std::cout << "Original tokens: ";
        for (int t : old_tokens) {
            std::cout << t << "(" << tiktoken_->decode({t}) << ") ";
        }
        std::cout << std::endl;
        
        // Convert to our new token IDs
        std::vector<int> new_tokens;
        new_tokens.reserve(old_tokens.size() + 2);
        new_tokens.push_back(tokens::BOS_ID);
        
        size_t unk_count = 0;
        for (int old_id : old_tokens) {
            int new_id = convert_to_new_id(old_id);
            if (new_id == tokens::UNK_ID) {
                unk_count++;
                std::cout << "Token " << old_id << "(" << tiktoken_->decode({old_id}) << ") mapped to UNK" << std::endl;
            }
            new_tokens.push_back(new_id);
        }
        
        new_tokens.push_back(tokens::EOS_ID);
        
        if (unk_count > 0) {
            std::cout << "Replaced " << unk_count << " tokens with <unk> token" << std::endl;
        }
        
        return new_tokens;
    } catch (const std::exception& e) {
        std::cout << "Error encoding text: '" << text << "': " << e.what() << std::endl;
        throw std::runtime_error("Encoding failed: " + std::string(e.what()));
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
    // Return the number of tokens in our new vocabulary (including special tokens)
    return old_to_new_id_.size() + 5;  // +5 for special tokens
} 