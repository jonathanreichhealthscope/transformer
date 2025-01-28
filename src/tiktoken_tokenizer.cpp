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
std::vector<std::string> extract_phrases(const std::string& filepath) {
    std::vector<std::string> phrases;
    std::ifstream file(filepath);
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        size_t sep_pos = line.find('|');
        if (sep_pos != std::string::npos) {
            // Get both parts for vocabulary building
            std::string context = line.substr(0, sep_pos);
            std::string target = line.substr(sep_pos + 1);
            
            // Add both parts if they're not empty
            if (!context.empty()) phrases.push_back(context);
            if (!target.empty()) phrases.push_back(target);
        }
    }
    return phrases;
}

void TiktokenTokenizer::initialize(const std::string& encoding_name) {
    try {
        std::cout << "Initializing custom tokenizer..." << std::endl;
        
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
        
        // Extract all phrases
        auto training_phrases = extract_phrases(training_path.string());
        auto validation_phrases = extract_phrases(validation_path.string());
        
        // Combine all phrases
        std::vector<std::string> all_phrases;
        all_phrases.insert(all_phrases.end(), training_phrases.begin(), training_phrases.end());
        all_phrases.insert(all_phrases.end(), validation_phrases.begin(), validation_phrases.end());
        
        std::cout << "Extracted " << all_phrases.size() << " total phrases" << std::endl;
        
        // Initialize vocabulary with special tokens
        std::vector<std::string> vocab = {
            "<pad>", "<unk>", "<s>", "</s>", "<mask>"
        };
        
        // First add all individual characters
        std::unordered_set<char> chars;
        for (const auto& phrase : all_phrases) {
            for (char c : phrase) {
                chars.insert(c);
            }
        }
        
        // Add individual characters to vocab
        for (char c : chars) {
            if (c != '|') {  // Skip the separator character
                vocab.push_back(std::string(1, c));
            }
        }
        
        // Now add all complete words as tokens
        std::unordered_map<std::string, int> word_freq;
        std::regex word_pattern(R"([a-zA-Z0-9]+(?:['-][a-zA-Z0-9]+)*|[.,!?;])");
        
        for (const auto& phrase : all_phrases) {
            auto words_begin = std::sregex_iterator(phrase.begin(), phrase.end(), word_pattern);
            auto words_end = std::sregex_iterator();
            
            for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
                std::string word = i->str();
                word_freq[word]++;
            }
        }
        
        // Sort words by frequency
        std::vector<std::pair<std::string, int>> sorted_words(word_freq.begin(), word_freq.end());
        std::sort(sorted_words.begin(), sorted_words.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Add most frequent words to vocabulary
        for (const auto& [word, freq] : sorted_words) {
            if (vocab.size() >= target_vocab_size) break;
            vocab.push_back(word);
        }
        
        // Now add common bigrams and trigrams if we still have space
        if (vocab.size() < target_vocab_size) {
            std::unordered_map<std::string, int> ngram_freq;
            
            for (const auto& phrase : all_phrases) {
                auto words_begin = std::sregex_iterator(phrase.begin(), phrase.end(), word_pattern);
                auto words_end = std::sregex_iterator();
                
                std::vector<std::string> words;
                for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
                    words.push_back(i->str());
                }
                
                // Add bigrams and trigrams
                for (size_t i = 0; i < words.size(); i++) {
                    if (i + 1 < words.size()) {
                        std::string bigram = words[i] + " " + words[i + 1];
                        ngram_freq[bigram]++;
                    }
                    if (i + 2 < words.size()) {
                        std::string trigram = words[i] + " " + words[i + 1] + " " + words[i + 2];
                        ngram_freq[trigram]++;
                    }
                }
            }
            
            // Sort n-grams by frequency
            std::vector<std::pair<std::string, int>> sorted_ngrams(ngram_freq.begin(), ngram_freq.end());
            std::sort(sorted_ngrams.begin(), sorted_ngrams.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
            
            // Add most frequent n-grams
            for (const auto& [ngram, freq] : sorted_ngrams) {
                if (vocab.size() >= target_vocab_size) break;
                vocab.push_back(ngram);
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
        std::cout << "- Regular tokens: " << (vocab.size() - 5) << std::endl;
        
        // Print some example tokens
        std::cout << "\nExample tokens from vocabulary:" << std::endl;
        for (size_t i = 5; i < std::min(vocab.size(), size_t(15)); i++) {
            std::cout << std::setw(3) << i << ": '" << vocab[i] << "'" << std::endl;
        }
        
        // Print top 10 most frequent words
        std::cout << "\nTop 10 most frequent words:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(10), sorted_words.size()); i++) {
            const auto& [word, freq] = sorted_words[i];
            std::cout << std::setw(3) << (i + 1) << ". '" << word << "': " 
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