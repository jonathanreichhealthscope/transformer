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
        
        // First, collect all GPT2 tokens and their IDs with frequency tracking
        std::vector<std::pair<std::string, int>> gpt2_tokens;
        std::unordered_map<int, size_t> token_counts;  // Track frequency of each token
        size_t total_tokens_processed = 0;
        
        // Create a set of unique token IDs, excluding UNK token
        std::unordered_set<int> seen_ids;
        
        // First pass: analyze training data from validation_pairs.txt to get actual token usage
        // Try multiple possible locations for the file
        std::filesystem::path data_file;
        std::vector<std::filesystem::path> possible_paths = {
            "data/validation_pairs.txt",
            "../data/validation_pairs.txt",
            "../../data/validation_pairs.txt",
            std::filesystem::current_path() / "data/validation_pairs.txt",
            std::filesystem::current_path() / "../data/validation_pairs.txt"
        };
        
        std::ifstream training_file;
        for (const auto& path : possible_paths) {
            if (std::filesystem::exists(path)) {
                data_file = path;
                training_file.open(path);
                if (training_file.is_open()) {
                    std::cout << "Found training file at: " << path << std::endl;
                    break;
                }
            }
        }
        
        if (!training_file.is_open()) {
            std::stringstream error_msg;
            error_msg << "Could not open training file. Tried the following paths:\n";
            for (const auto& path : possible_paths) {
                error_msg << "- " << path << "\n";
            }
            error_msg << "Current working directory: " << std::filesystem::current_path() << "\n";
            throw std::runtime_error(error_msg.str());
        }
        
        std::string line;
        while (std::getline(training_file, line)) {
            // Skip empty lines
            if (line.empty()) continue;
            
            // Encode the line to see what tokens are actually used
            std::vector<int> ids = tiktoken_->encode(line);
            total_tokens_processed += ids.size();
            
            for (const auto& id : ids) {
                token_counts[id]++;
                
                if (seen_ids.insert(id).second) {  // If this is a new ID
                    std::string token = tiktoken_->decode({id});
                    gpt2_tokens.push_back({token, id});
                }
            }
        }
        
        // Print initial statistics
        std::cout << "\nToken Usage Statistics:" << std::endl;
        std::cout << "- Total tokens processed: " << total_tokens_processed << std::endl;
        std::cout << "- Unique tokens found: " << seen_ids.size() << std::endl;
        
        // Add common programming and special characters if not already seen
        std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-_'\"();:$/ \n\t";
        for (char c : chars) {
            std::string token_str(1, c);
            std::vector<int> ids = tiktoken_->encode(token_str);
            for (const auto& id : ids) {
                if (seen_ids.insert(id).second) {
                    std::string token = tiktoken_->decode({id});
                    gpt2_tokens.push_back({token, id});
                    token_counts[id] = 1;  // Give it a small count if not seen in training
                }
            }
        }
        
        // Sort tokens by frequency, then by GPT2 ID for ties
        std::sort(gpt2_tokens.begin(), gpt2_tokens.end(),
                 [&token_counts](const auto& a, const auto& b) {
                     if (token_counts[a.second] != token_counts[b.second]) {
                         return token_counts[a.second] > token_counts[b.second];
                     }
                     return a.second < b.second;
                 });
        
        // Create our token mappings
        old_to_new_id_.clear();
        new_to_old_id_.clear();
        
        // First, map special tokens
        setup_special_tokens();
        for (int i = 0; i < 5; i++) {
            old_to_new_id_[i] = i;
            new_to_old_id_[i] = i;
        }
        
        // Ensure target_vocab_size is at least as large as our unique tokens
        size_t min_required_size = seen_ids.size() + 5;  // Add 5 for special tokens
        if (target_vocab_size < min_required_size) {
            std::cout << "\nWarning: Increasing target_vocab_size from " << target_vocab_size 
                      << " to " << min_required_size << " to accommodate all unique tokens" << std::endl;
            target_vocab_size = min_required_size;
        }
        
        // Then map GPT2 tokens to our consecutive IDs, prioritizing by frequency
        int current_id = 5;  // Start after special tokens
        size_t total_occurrences = 0;
        size_t mapped_occurrences = 0;
        
        // First calculate total occurrences (excluding special tokens and UNK)
        for (const auto& [token, gpt2_id] : gpt2_tokens) {
            if (gpt2_id >= 5) {  // Skip special tokens
                total_occurrences += token_counts[gpt2_id];
            }
        }
        
        // Map tokens until we hit target_vocab_size
        std::vector<std::pair<std::string, size_t>> token_frequency_list;
        for (const auto& [token, gpt2_id] : gpt2_tokens) {
            if (current_id >= target_vocab_size) break;
            
            // Skip if this is a special token
            if (gpt2_id < 5) continue;
            
            old_to_new_id_[gpt2_id] = current_id;
            new_to_old_id_[current_id] = gpt2_id;
            mapped_occurrences += token_counts[gpt2_id];
            
            // Store for frequency reporting
            token_frequency_list.push_back({token, token_counts[gpt2_id]});
            
            current_id++;
        }
        
        std::cout << "\nVocabulary mapping complete:" << std::endl;
        std::cout << "- Total mapped tokens: " << current_id << std::endl;
        std::cout << "- Special tokens: 5" << std::endl;
        std::cout << "- Regular GPT-2 tokens: " << (current_id - 5) << std::endl;
        std::cout << "- Total GPT-2 vocabulary size: " << tiktoken_->get_vocab_size() << std::endl;
        std::cout << "- Coverage of training data: " 
                  << std::fixed << std::setprecision(2)
                  << (100.0 * mapped_occurrences / total_occurrences) << "%" << std::endl;
        
        // Print top 10 most frequent tokens
        std::cout << "\nTop 10 most frequent tokens:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(10), token_frequency_list.size()); i++) {
            const auto& [token, count] = token_frequency_list[i];
            std::cout << std::setw(3) << (i + 1) << ". '" << token << "': " 
                      << count << " occurrences (" 
                      << std::fixed << std::setprecision(2)
                      << (100.0 * count / total_occurrences) << "%)" << std::endl;
        }
        
        // Initialize token frequencies based on actual usage
        token_frequencies_.clear();
        for (const auto& [token, gpt2_id] : gpt2_tokens) {
            if (old_to_new_id_.find(gpt2_id) != old_to_new_id_.end()) {
                token_frequencies_[token] = static_cast<float>(token_counts[gpt2_id]) / total_occurrences;
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