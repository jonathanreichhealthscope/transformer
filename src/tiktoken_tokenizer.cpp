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
        
        // Initialize the base tiktoken encoder with GPT-2 encoding
        tiktoken_ = std::make_unique<tiktoken::Encoding>("gpt2");
        std::cout << "Loaded base GPT-2 vocabulary" << std::endl;
        
        // Get the executable's directory
        std::filesystem::path exe_path = std::filesystem::current_path();
        std::unordered_map<int, size_t> id_freqs;
        
        // Function to process a data file and update frequencies
        auto process_file = [&](const std::string& filename) -> bool {
            std::ifstream file;
            std::string file_path;
            
            // First try data directory relative to executable
            auto data_path = exe_path / "data" / filename;
            file.open(data_path);
            if (file.is_open()) {
                file_path = data_path.string();
            } else {
                // Try parent directory
                data_path = exe_path.parent_path() / "data" / filename;
                file.open(data_path);
                if (file.is_open()) {
                    file_path = data_path.string();
                } else {
                    std::cerr << "ERROR: Could not find " << filename << " in either:\n";
                    std::cerr << "  - " << (exe_path / "data" / filename).string() << "\n";
                    std::cerr << "  - " << (exe_path.parent_path() / "data" / filename).string() << "\n";
                    return false;
                }
            }

            std::cout << "Processing " << filename << " from: " << file_path << std::endl;

            // Count total lines for progress reporting
            size_t total_lines = 0;
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty()) total_lines++;
            }
            
            // Reset file to beginning
            file.clear();
            file.seekg(0);

            // Process data to build token frequencies
            size_t processed_lines = 0;
            std::string buffer;
            
            std::cout << "\nProcessing " << filename << ":\n";
            const int bar_width = 50;
            
            // Read file in larger chunks for efficiency
            while (std::getline(file, line)) {
                if (line.empty()) continue;
                
                buffer += line + "\n";
                processed_lines++;
                
                // Process buffer when it gets large enough
                if (buffer.length() > 10000 || processed_lines == total_lines) {
                    std::vector<int> token_ids = tiktoken_->encode(buffer);
                    for (int id : token_ids) {
                        id_freqs[id]++;
                    }
                    buffer.clear();
                }
                
                // Update progress bar
                float progress = float(processed_lines) / total_lines;
                int pos = bar_width * progress;
                
                std::cout << "\r[";
                for (int i = 0; i < bar_width; ++i) {
                    if (i < pos) std::cout << "=";
                    else if (i == pos) std::cout << ">";
                    else std::cout << " ";
                }
                std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0) << "% "
                         << "(" << processed_lines << "/" << total_lines << ")" << std::flush;
            }
            std::cout << std::endl;
            return true;
        };

        // Process both training and validation data
        if (!process_file("training_pairs.txt")) return;
        if (!process_file("validation_pairs.txt")) return;
        
        // Add test queries to vocabulary
        std::cout << "\nProcessing test queries..." << std::endl;
        std::vector<std::string> test_queries = {
            "I go to",
            "The weather is",
            "I want to",
            "The cat",
            "She likes to"
        };
        
        for (const auto& query : test_queries) {
            std::vector<int> token_ids = tiktoken_->encode(query);
            for (int id : token_ids) {
                id_freqs[id]++;
            }
        }
        
        // Convert ID frequencies to token frequencies
        std::cout << "\nConverting token frequencies..." << std::endl;
        for (const auto& [id, freq] : id_freqs) {
            std::string token = tiktoken_->decode({id});
            if (!token.empty()) {
                token_frequencies_[token] = freq;
            }
        }
        
        std::cout << "Building vocabulary from frequencies..." << std::endl;
        build_vocabulary_from_frequencies();
        
        std::cout << "Setting up special tokens..." << std::endl;
        setup_special_tokens();
        
        std::cout << "Final vocabulary breakdown:" << std::endl;
        std::cout << "- Special tokens: 5" << std::endl;
        std::cout << "- Regular tokens: " << (old_to_new_id_.size() - 5) << std::endl;
        std::cout << "Total vocabulary size: " << vocab_size() << " tokens" << std::endl;
        
        // Print some statistics about the vocabulary
        size_t total_occurrences = 0;
        for (const auto& [token, freq] : token_frequencies_) {
            total_occurrences += freq;
        }
        std::cout << "Total token occurrences: " << total_occurrences << std::endl;
        std::cout << "Unique tokens before filtering: " << token_frequencies_.size() << std::endl;
        std::cout << "Tokenizer initialization complete!" << std::endl;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize tokenizer: " + std::string(e.what()));
    }
}

void TiktokenTokenizer::build_vocabulary_from_frequencies() {
    // Create vector of token-frequency pairs
    std::vector<std::pair<std::string, size_t>> freq_pairs(token_frequencies_.begin(), token_frequencies_.end());
    
    // Sort by frequency, highest first
    std::sort(freq_pairs.begin(), freq_pairs.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Take top tokens up to target_vocab_size
    size_t current_id = 5;  // Start after special tokens
    filtered_tokens_.resize(tiktoken_->get_vocab_size(), false);
    
    // Debug info
    if (debug_logging_) {
        std::cout << "Top 10 most frequent tokens:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(10), freq_pairs.size()); i++) {
            std::cout << freq_pairs[i].first << ": " << freq_pairs[i].second << std::endl;
        }
    }
    
    for (const auto& [token, freq] : freq_pairs) {
        if (current_id >= target_vocab_size) break;
        
        // Get original token ID from tiktoken
        std::vector<int> token_ids = tiktoken_->encode(token);
        if (token_ids.empty()) continue;
        
        // Only add single tokens to maintain integrity
        if (token_ids.size() == 1) {
            int old_id = token_ids[0];
            filtered_tokens_[old_id] = true;
            old_to_new_id_[old_id] = current_id;
            new_to_old_id_[current_id] = old_id;
            current_id++;
        }
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
        // Convert our new token IDs back to original IDs
        std::vector<int> old_tokens;
        old_tokens.reserve(new_tokens.size());
        
        for (int new_id : new_tokens) {
            old_tokens.push_back(convert_to_old_id(new_id));
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