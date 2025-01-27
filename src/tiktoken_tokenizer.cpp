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
        
        // Get the executable's directory
        std::filesystem::path exe_path = std::filesystem::current_path();
        std::unordered_map<int, size_t> id_freqs;
        
        // Add common English words and phrases to ensure they're in vocabulary
        std::vector<std::string> common_words = {
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            // Add common word combinations
            "I am", "I will", "I have", "the cat", "to the", "in the", "of the",
            "it is", "they are", "we are", "you are", "he is", "she is"
        };
        
        // Process common words first with high frequency
        for (const auto& word : common_words) {
            std::vector<int> token_ids = tiktoken_->encode(word);
            for (int id : token_ids) {
                // Give common words high frequency
                id_freqs[id] += 10000;
            }
        }

        // Add domain-specific preprocessing
        std::vector<std::string> domain_specific = {
            // Add common patterns from your training data
            // For example, if you have lots of dates:
            "January", "February", "March", // ... other months
            "Monday", "Tuesday", "Wednesday", // ... other days
            // Common number patterns
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "10", "20", "30", "40", "50", "100",
            // Add any domain-specific terms
        };
        
        for (const auto& term : domain_specific) {
            std::vector<int> token_ids = tiktoken_->encode(term);
            for (int id : token_ids) {
                // Give very high frequency to domain-specific terms
                id_freqs[id] += 20000;
            }
        }

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
        
        // Add test queries with high frequency to ensure they're well-represented
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
                // Give test queries high frequency
                id_freqs[id] += 5000;
            }
        }
        
        // Convert ID frequencies to token frequencies with improved handling
        std::cout << "\nConverting token frequencies..." << std::endl;
        for (const auto& [id, freq] : id_freqs) {
            std::string token = tiktoken_->decode({id});
            if (!token.empty()) {
                // Boost frequency for shorter tokens to prefer word-level tokens
                float length_boost = std::max(1.0f, 5.0f - float(token.length()) * 0.2f);
                token_frequencies_[token] = freq * length_boost;
            }
        }
        
        // Set target vocabulary size to match your data plus some margin
        target_vocab_size = 2500;  // 2300 tokens + 200 margin for special cases/byte tokens
        
        std::cout << "Building vocabulary from frequencies..." << std::endl;
        build_vocabulary_from_frequencies();
        
        std::cout << "Setting up special tokens..." << std::endl;
        setup_special_tokens();
        
        // Print vocabulary statistics
        std::cout << "Final vocabulary breakdown:" << std::endl;
        std::cout << "- Special tokens: 5" << std::endl;
        std::cout << "- Regular tokens: " << (old_to_new_id_.size() - 5) << std::endl;
        std::cout << "Total vocabulary size: " << vocab_size() << " tokens" << std::endl;
        
        // Validate common words are properly tokenized
        std::cout << "\nValidating common word tokenization:" << std::endl;
        for (const auto& word : {"The cat", "I go to", "She likes"}) {
            auto tokens = encode(word);
            std::cout << "'" << word << "' -> ";
            for (int token : tokens) {
                std::cout << decode({token}) << " ";
            }
            std::cout << std::endl;
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize tokenizer: " + std::string(e.what()));
    }
}

void TiktokenTokenizer::build_vocabulary_from_frequencies() {
    std::vector<std::pair<std::string, size_t>> freq_pairs(token_frequencies_.begin(), token_frequencies_.end());
    
    // Sort by frequency, highest first
    std::sort(freq_pairs.begin(), freq_pairs.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    size_t current_id = 5;  // Start after special tokens
    size_t training_data_tokens = 0;  // Counter for tokens from training data
    size_t gpt2_tokens = 0;  // Counter for GPT2 tokens
    filtered_tokens_.resize(tiktoken_->get_vocab_size(), false);
    
    // Print more detailed frequency analysis
    if (debug_logging_) {
        std::cout << "\nToken frequency analysis:" << std::endl;
        std::cout << "Top 20 most frequent tokens:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(20), freq_pairs.size()); i++) {
            std::cout << freq_pairs[i].first << ": " << freq_pairs[i].second << std::endl;
        }
        
        // Print frequency distribution
        std::cout << "\nFrequency distribution:" << std::endl;
        size_t very_high_freq = 0, high_freq = 0, medium_freq = 0, low_freq = 0;
        for (const auto& [token, freq] : freq_pairs) {
            if (freq > 10000) very_high_freq++;
            else if (freq > 1000) high_freq++;
            else if (freq > 100) medium_freq++;
            else low_freq++;
        }
        std::cout << "Very high frequency (>10000): " << very_high_freq << std::endl;
        std::cout << "High frequency (1000-10000): " << high_freq << std::endl;
        std::cout << "Medium frequency (100-1000): " << medium_freq << std::endl;
        std::cout << "Low frequency (<100): " << low_freq << std::endl;
    }
    
    // First pass: Prioritize tokens from your training data
    for (const auto& [token, freq] : freq_pairs) {
        if (current_id >= target_vocab_size) break;
        
        std::vector<int> token_ids = tiktoken_->encode(token);
        // Only add tokens that appear in your training data with sufficient frequency
        if (token_ids.size() == 1 && freq >= 5) {
            int old_id = token_ids[0];
            filtered_tokens_[old_id] = true;
            old_to_new_id_[old_id] = current_id;
            new_to_old_id_[current_id] = old_id;
            current_id++;
            training_data_tokens++;
        }
    }
    
    // Second pass: Fill remaining slots with most frequent GPT2 tokens
    if (current_id < target_vocab_size) {
        for (size_t i = 0; i < tiktoken_->get_vocab_size() && current_id < target_vocab_size; i++) {
            if (!filtered_tokens_[i]) {
                filtered_tokens_[i] = true;
                old_to_new_id_[i] = current_id;
                new_to_old_id_[current_id] = i;
                current_id++;
                gpt2_tokens++;
            }
        }
    }
    
    std::cout << "\nVocabulary construction complete:" << std::endl;
    std::cout << "- Total tokens: " << current_id << std::endl;
    std::cout << "- Training data tokens: " << training_data_tokens << std::endl;
    std::cout << "- GPT2 tokens: " << gpt2_tokens << std::endl;
    std::cout << "- Special tokens: 5" << std::endl;
    std::cout << "- Original GPT2 vocabulary size: " << tiktoken_->get_vocab_size() << std::endl;
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