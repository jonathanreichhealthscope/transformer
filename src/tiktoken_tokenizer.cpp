#include "../include/tiktoken_tokenizer.hpp"
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <filesystem>
#include <regex>
#include <iomanip>

TiktokenTokenizer::TiktokenTokenizer() = default;

// Helper to get all complete phrases (targets) from the data
std::vector<std::pair<std::string, std::string>> extract_phrase_pairs(const std::string& filepath) {
    std::vector<std::pair<std::string, std::string>> pairs;
    std::ifstream file(filepath);
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        size_t sep_pos = line.find('|');
        if (sep_pos != std::string::npos) {
            std::string context = line.substr(0, sep_pos);
            std::string target = line.substr(sep_pos + 1);
            
            // Only trim the context, preserve exact target format
            context.erase(0, context.find_first_not_of(" \t\r\n"));
            context.erase(context.find_last_not_of(" \t\r\n") + 1);
            
            // Add a space prefix to target if it doesn't have one
            if (!target.empty() && target[0] != ' ') {
                target = " " + target;
            }
            
            if (!context.empty() && !target.empty()) {
                pairs.emplace_back(context, target);
            }
        }
    }
    return pairs;
}

void TiktokenTokenizer::initialize(const std::string& encoding_name) {
    try {
        std::cout << "Initializing custom tokenizer for noun phrase completion..." << std::endl;
        
        // Find our data files
        std::vector<std::filesystem::path> possible_paths = {
            "data/training_pairs.txt",
            "../data/training_pairs.txt",
            "../../data/training_pairs.txt",
            std::filesystem::current_path() / "data/training_pairs.txt",
            std::filesystem::current_path() / "../data/training_pairs.txt"
        };
        
        std::filesystem::path training_path;
        for (const auto& path : possible_paths) {
            if (std::filesystem::exists(path)) {
                training_path = path;
                break;
            }
        }
        
        if (training_path.empty()) {
            throw std::runtime_error("Could not find training data file");
        }
        
        // Get validation file path
        auto validation_path = training_path.parent_path() / "validation_pairs.txt";
        if (!std::filesystem::exists(validation_path)) {
            throw std::runtime_error("Could not find validation data file");
        }
        
        std::cout << "Found data files:\n"
                  << "- Training: " << training_path << "\n"
                  << "- Validation: " << validation_path << std::endl;
        
        // Extract all phrase pairs
        auto training_pairs = extract_phrase_pairs(training_path.string());
        auto validation_pairs = extract_phrase_pairs(validation_path.string());
        
        // Collect target phrases and their frequencies
        std::unordered_map<std::string, int> target_freq;
        std::vector<std::string> all_targets;
        
        for (const auto& [context, target] : training_pairs) {
            target_freq[target]++;
            all_targets.push_back(target);
        }
        for (const auto& [context, target] : validation_pairs) {
            target_freq[target]++;
            all_targets.push_back(target);
        }
        
        std::cout << "Extracted " << all_targets.size() << " total phrases" << std::endl;
        std::cout << "Found " << target_freq.size() << " unique target phrases" << std::endl;
        
        // Initialize vocabulary with special tokens
        std::vector<std::string> vocab = {
            "<pad>", "<unk>", "<s>", "</s>", "<mask>"
        };
        
        // Sort target phrases by frequency
        std::vector<std::pair<std::string, int>> sorted_targets(target_freq.begin(), target_freq.end());
        std::sort(sorted_targets.begin(), sorted_targets.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // First add complete target phrases as tokens (with their spaces)
        for (const auto& [phrase, freq] : sorted_targets) {
            if (vocab.size() >= target_vocab_size) break;
            
            // Ensure the phrase starts with a space
            std::string token = phrase;
            if (!token.empty() && token[0] != ' ') {
                token = " " + token;
            }
            vocab.push_back(token);
        }
        
        // If we still have space, add individual words from targets (with spaces)
        if (vocab.size() < target_vocab_size) {
            std::unordered_map<std::string, int> word_freq;
            std::regex word_pattern(R"(\s*([a-zA-Z0-9]+(?:['-][a-zA-Z0-9]+)*|[.,!?;]))");
            
            for (const auto& [phrase, _] : sorted_targets) {
                auto words_begin = std::sregex_iterator(phrase.begin(), phrase.end(), word_pattern);
                auto words_end = std::sregex_iterator();
                
                for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
                    std::string word = " " + i->str();  // Add space prefix
                    word_freq[word]++;
                }
            }
            
            // Sort words by frequency
            std::vector<std::pair<std::string, int>> sorted_words(word_freq.begin(), word_freq.end());
            std::sort(sorted_words.begin(), sorted_words.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
            
            // Add most frequent words (they already have space prefixes)
            for (const auto& [word, freq] : sorted_words) {
                if (vocab.size() >= target_vocab_size) break;
                vocab.push_back(word);
            }
        }
        
        // Clear existing mappings
        token_to_id_.clear();
        id_to_token_.clear();
        
        // Create the token mappings
        for (size_t i = 0; i < vocab.size(); i++) {
            token_to_id_[vocab[i]] = i;
            id_to_token_[i] = vocab[i];
        }
        
        std::cout << "\nVocabulary statistics:" << std::endl;
        std::cout << "- Total vocabulary size: " << vocab.size() << std::endl;
        std::cout << "- Special tokens: 5" << std::endl;
        std::cout << "- Complete phrases: " << std::min(sorted_targets.size(), target_vocab_size - 5) << std::endl;
        std::cout << "- Individual words: " << (vocab.size() - 5 - std::min(sorted_targets.size(), target_vocab_size - 5)) << std::endl;
        
        // Print example complete phrases
        std::cout << "\nTop 10 most common target phrases:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(10), sorted_targets.size()); i++) {
            const auto& [phrase, freq] = sorted_targets[i];
            std::cout << std::setw(3) << (i + 1) << ". '" << phrase << "': " 
                      << freq << " occurrences" << std::endl;
        }
        
        is_initialized_ = true;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize tokenizer: " + std::string(e.what()));
    }
}

std::vector<int> TiktokenTokenizer::encode(const std::string& text) const {
    if (!is_initialized_) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    
    std::vector<int> tokens;
    std::string remaining = text;
    
    while (!remaining.empty()) {
        size_t best_len = 0;
        int best_token = tokens::UNK_ID;
        
        // Try to match the longest token possible
        for (const auto& [token, id] : token_to_id_) {
            if (token.length() > remaining.length()) continue;
            
            if (remaining.substr(0, token.length()) == token) {
                if (token.length() > best_len) {
                    best_len = token.length();
                    best_token = id;
                }
            }
        }
        
        // Add the best token found (or UNK if none found)
        tokens.push_back(best_token);
        
        // Remove the matched portion
        if (best_len > 0) {
            remaining = remaining.substr(best_len);
        } else {
            // If no match found, skip one character
            remaining = remaining.substr(1);
        }
    }
    
    return tokens;
}

std::string TiktokenTokenizer::decode(const std::vector<int>& tokens) const {
    if (!is_initialized_) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    
    std::string result;
    bool first_token = true;
    
    for (int token_id : tokens) {
        auto it = id_to_token_.find(token_id);
        if (it != id_to_token_.end()) {
            if (!first_token && it->second[0] != ' ' && !result.empty() && result.back() != ' ') {
                result += ' ';
            }
            result += it->second;
        } else {
            result += "<unk>";
        }
        first_token = false;
    }
    
    return result;
}

size_t TiktokenTokenizer::vocab_size() const {
    return token_to_id_.size();
} 