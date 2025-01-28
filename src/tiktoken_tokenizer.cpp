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
#include <nlohmann/json.hpp>
#include <algorithm>

TiktokenTokenizer::TiktokenTokenizer() = default;

// Helper to identify if a phrase ends with an adjective
bool is_adjective_ending(const std::string& phrase) {
    static const std::unordered_set<std::string> common_adjective_endings = {
        "able", "ible", "al", "ful", "ic", "ive", "less", "ous", "y"
    };
    
    // Common adjectives that don't follow standard patterns
    static const std::unordered_set<std::string> common_adjectives = {
        " bright", " dark", " hot", " cold", " big", " small", " tall", " short",
        " red", " blue", " green", " black", " white", " yellow", " good", " bad",
        " fast", " slow", " hard", " soft", " loud", " quiet", " rich", " poor",
        " young", " old", " new", " old", " happy", " sad", " clean", " dirty"
    };
    
    // First check if it's a common adjective
    if (common_adjectives.find(phrase) != common_adjectives.end()) {
        return true;
    }
    
    // Then check for adjective endings
    for (const auto& ending : common_adjective_endings) {
        if (phrase.length() > ending.length() && 
            phrase.substr(phrase.length() - ending.length()) == ending) {
            return true;
        }
    }
    
    return false;
}

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
        
        // Collect target phrases and their frequencies, separating adjectives
        std::unordered_map<std::string, int> adjective_freq;
        std::unordered_map<std::string, int> other_target_freq;
        std::vector<std::string> all_targets;
        
        // First pass: separate adjectives and other targets
        for (const auto& [context, target] : training_pairs) {
            if (is_adjective_ending(target)) {
                adjective_freq[target]++;
            } else {
                other_target_freq[target]++;
            }
            all_targets.push_back(target);
        }
        for (const auto& [context, target] : validation_pairs) {
            if (is_adjective_ending(target)) {
                adjective_freq[target]++;
            } else {
                other_target_freq[target]++;
            }
            all_targets.push_back(target);
        }
        
        std::cout << "Extracted " << all_targets.size() << " total phrases" << std::endl;
        std::cout << "Found " << adjective_freq.size() << " unique adjective phrases" << std::endl;
        std::cout << "Found " << other_target_freq.size() << " other unique phrases" << std::endl;
        
        // Initialize vocabulary with special tokens
        std::vector<std::string> vocab = {
            "<pad>", "<unk>", "<s>", "</s>", "<mask>", "|"
        };
        
        // Sort adjectives by frequency
        std::vector<std::pair<std::string, int>> sorted_adjectives(adjective_freq.begin(), adjective_freq.end());
        std::sort(sorted_adjectives.begin(), sorted_adjectives.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Sort other targets by frequency
        std::vector<std::pair<std::string, int>> sorted_others(other_target_freq.begin(), other_target_freq.end());
        std::sort(sorted_others.begin(), sorted_others.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Reserve space for adjectives (ensure they get good token IDs)
        size_t adjective_quota = std::min(size_t(target_vocab_size * 0.3), sorted_adjectives.size());
        
        // Add adjectives first to ensure they get good token IDs
        for (const auto& [phrase, freq] : sorted_adjectives) {
            if (vocab.size() >= tokens::MASK_ID + 1 + adjective_quota) break;
            
            // Ensure the phrase starts with a space
            std::string token = phrase;
            if (!token.empty() && token[0] != ' ') {
                token = " " + token;
            }
            vocab.push_back(token);
        }
        
        // Then add other frequent phrases
        for (const auto& [phrase, freq] : sorted_others) {
            if (vocab.size() >= target_vocab_size) break;
            
            std::string token = phrase;
            if (!token.empty() && token[0] != ' ') {
                token = " " + token;
            }
            vocab.push_back(token);
        }
        
        // If we still have space, add individual words
        if (vocab.size() < target_vocab_size) {
            std::unordered_map<std::string, int> word_freq;
            std::regex word_pattern(R"(\s*([a-zA-Z0-9]+(?:['-][a-zA-Z0-9]+)*|[.,!?;]))");
            
            // Prioritize words from adjective phrases
            for (const auto& [phrase, _] : sorted_adjectives) {
                auto words_begin = std::sregex_iterator(phrase.begin(), phrase.end(), word_pattern);
                auto words_end = std::sregex_iterator();
                
                for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
                    std::string word = " " + i->str();
                    word_freq[word] += 2;  // Give higher weight to adjective words
                }
            }
            
            // Then add words from other phrases
            for (const auto& [phrase, _] : sorted_others) {
                auto words_begin = std::sregex_iterator(phrase.begin(), phrase.end(), word_pattern);
                auto words_end = std::sregex_iterator();
                
                for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
                    std::string word = " " + i->str();
                    word_freq[word]++;
                }
            }
            
            // Sort words by frequency
            std::vector<std::pair<std::string, int>> sorted_words(word_freq.begin(), word_freq.end());
            std::sort(sorted_words.begin(), sorted_words.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
            
            // Add remaining words
            for (const auto& [word, freq] : sorted_words) {
                if (vocab.size() >= target_vocab_size) break;
                vocab.push_back(word);
            }
        }
        
        // Create token mappings (rest of the initialization remains the same)
        token_to_id_.clear();
        id_to_token_.clear();
        
        // Add special tokens with fixed IDs
        token_to_id_["<pad>"] = tokens::PAD_ID;
        token_to_id_["<unk>"] = tokens::UNK_ID;
        token_to_id_["<s>"] = tokens::BOS_ID;
        token_to_id_["</s>"] = tokens::EOS_ID;
        token_to_id_["<mask>"] = tokens::MASK_ID;
        token_to_id_["|"] = tokens::SEP_ID;
        
        id_to_token_[tokens::PAD_ID] = "<pad>";
        id_to_token_[tokens::UNK_ID] = "<unk>";
        id_to_token_[tokens::BOS_ID] = "<s>";
        id_to_token_[tokens::EOS_ID] = "</s>";
        id_to_token_[tokens::MASK_ID] = "<mask>";
        id_to_token_[tokens::SEP_ID] = "|";
        
        // Add vocabulary tokens with consecutive IDs
        int next_id = tokens::MASK_ID + 1;
        for (const auto& token : vocab) {
            if (token_to_id_.find(token) == token_to_id_.end()) {
                token_to_id_[token] = next_id;
                id_to_token_[next_id] = token;
                next_id++;
            }
        }
        
        vocab_size_ = id_to_token_.size();
        
        // Print statistics
        std::cout << "\nVocabulary statistics:" << std::endl;
        std::cout << "- Total vocabulary size: " << vocab_size_ << std::endl;
        std::cout << "- Special tokens: 6" << std::endl;
        std::cout << "- Adjective phrases: " << std::min(adjective_quota, sorted_adjectives.size()) << std::endl;
        std::cout << "- Other phrases: " << std::min(sorted_others.size(), target_vocab_size - 6 - adjective_quota) << std::endl;
        std::cout << "- Individual words: " << (vocab_size_ - 6 - std::min(adjective_quota, sorted_adjectives.size()) 
                                              - std::min(sorted_others.size(), target_vocab_size - 6 - adjective_quota)) << std::endl;
        
        // Print most common adjectives
        std::cout << "\nTop 10 most common adjective phrases:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(10), sorted_adjectives.size()); i++) {
            const auto& [phrase, freq] = sorted_adjectives[i];
            std::cout << std::setw(3) << (i + 1) << ". '" << phrase << "': " 
                      << freq << " occurrences" << std::endl;
        }
        
        // Print most common other phrases
        std::cout << "\nTop 10 most common other phrases:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(10), sorted_others.size()); i++) {
            const auto& [phrase, freq] = sorted_others[i];
            std::cout << std::setw(3) << (i + 1) << ". '" << phrase << "': " 
                      << freq << " occurrences" << std::endl;
        }
        
        is_initialized_ = true;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize tokenizer: " + std::string(e.what()));
    }
}

std::vector<int> TiktokenTokenizer::encode(const std::string& text, bool add_special_tokens) const {
    std::vector<int> tokens;
    
    if (add_special_tokens) {
        tokens.push_back(tokens::BOS_ID);
    }
    
    // Handle separator token specially
    size_t sep_pos = text.find(SEP_TOKEN);
    if (sep_pos != std::string::npos) {
        // Encode text before separator
        std::string prefix = text.substr(0, sep_pos);
        auto prefix_tokens = tokenize_text(prefix);
        tokens.insert(tokens.end(), prefix_tokens.begin(), prefix_tokens.end());
        
        // Add separator token
        tokens.push_back(tokens::SEP_ID);
        
        // Encode text after separator
        std::string suffix = text.substr(sep_pos + 1);
        auto suffix_tokens = tokenize_text(suffix);
        tokens.insert(tokens.end(), suffix_tokens.begin(), suffix_tokens.end());
    } else {
        auto text_tokens = tokenize_text(text);
        tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());
    }
    
    if (add_special_tokens) {
        tokens.push_back(tokens::EOS_ID);
    }
    
    return tokens;
}

std::string TiktokenTokenizer::decode(const std::vector<int>& token_ids, bool skip_special_tokens) const {
    std::string result;
    bool after_separator = false;
    
    for (int token_id : token_ids) {
        // Skip special tokens if requested
        if (skip_special_tokens) {
            if (token_id <= tokens::MASK_ID) continue;
        }
        
        if (token_id == tokens::SEP_ID) {
            result += SEP_TOKEN;
            after_separator = true;
            continue;
        }
        
        std::string token = decode_token(token_id);
        
        // Preserve exact spacing after separator
        if (after_separator && !token.empty() && token[0] != ' ') {
            token = " " + token;
        }
        
        result += token;
    }
    
    return result;
}

// Helper function to tokenize text segments
std::vector<int> TiktokenTokenizer::tokenize_text(const std::string& text) const {
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
        
        // Add the best token found
        tokens.push_back(best_token);
        
        // Remove the matched portion
        if (best_len > 0) {
            remaining = remaining.substr(best_len);
        } else {
            // If no match found, skip one character and use UNK token
            remaining = remaining.substr(1);
        }
    }
    
    return tokens;
}

// Helper function to decode individual tokens
std::string TiktokenTokenizer::decode_token(int token_id) const {
    if (!is_initialized_) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    
    auto it = id_to_token_.find(token_id);
    if (it != id_to_token_.end()) {
        return it->second;
    }
    
    return "<unk>";  // Return unknown token string if not found
} 