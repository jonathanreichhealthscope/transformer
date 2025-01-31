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

// Private implementation class
struct TiktokenTokenizer::Impl {
    bool initialized_ = false;
    std::unordered_map<std::string, int> vocab;
    size_t vocab_size_ = 32000; // Default GPT-2 vocab size

    void initialize(const std::string& encoding_name) {
        // TODO: Implement actual tiktoken initialization
        initialized_ = true;
    }

    bool is_initialized() const { return initialized_; }

    std::vector<int> encode(const std::string& text) const {
        if (!initialized_) throw std::runtime_error("Tokenizer not initialized");
        // TODO: Implement actual encoding
        return {0}; // Placeholder
    }

    std::string decode(const std::vector<int>& tokens) const {
        if (!initialized_) throw std::runtime_error("Tokenizer not initialized");
        // TODO: Implement actual decoding
        return ""; // Placeholder
    }

    size_t vocab_size() const {
        if (!initialized_) throw std::runtime_error("Tokenizer not initialized");
        return vocab_size_;
    }
};

TiktokenTokenizer::TiktokenTokenizer(const std::string& encoding_name)
    : pimpl_(std::make_unique<Impl>()) {
    initialize(encoding_name);
}

TiktokenTokenizer::~TiktokenTokenizer() = default;

void TiktokenTokenizer::initialize(const std::string& encoding_name) {
    if (initialized_) return;
    
    pimpl_->initialize(encoding_name);
    initialized_ = true;
    initialize_token_categories();
}

bool TiktokenTokenizer::is_initialized() const {
    return initialized_ && pimpl_->is_initialized();
}

std::vector<int> TiktokenTokenizer::encode(const std::string& text) const {
    if (!is_initialized()) throw std::runtime_error("Tokenizer not initialized");
    return pimpl_->encode(text);
}

std::string TiktokenTokenizer::decode(const std::vector<int>& tokens) const {
    if (!is_initialized()) throw std::runtime_error("Tokenizer not initialized");
    return pimpl_->decode(tokens);
}

size_t TiktokenTokenizer::vocab_size() const {
    if (!is_initialized()) throw std::runtime_error("Tokenizer not initialized");
    return pimpl_->vocab_size();
}

void TiktokenTokenizer::print_vocabulary_mappings() const {
    if (!is_initialized()) {
        std::cerr << "Warning: Attempting to print mappings before tokenizer initialization" << std::endl;
        return;
    }

    std::cout << "Special Token IDs:\n"
              << "PAD: " << get_pad_token_id() << "\n"
              << "UNK: " << get_unk_token_id() << "\n"
              << "BOS: " << get_bos_token_id() << "\n"
              << "EOS: " << get_eos_token_id() << "\n"
              << "MASK: " << get_mask_token_id() << std::endl;
}

void TiktokenTokenizer::save(std::ostream& os) const {
    if (!is_initialized()) throw std::runtime_error("Tokenizer not initialized");
    
    // Write version
    uint32_t version = 1;
    os.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Write vocab size
    uint32_t vocab_size = static_cast<uint32_t>(pimpl_->vocab_size());
    os.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
    
    // Write vocabulary
    for (const auto& [token, id] : token_to_id_) {
        uint32_t token_length = static_cast<uint32_t>(token.length());
        os.write(reinterpret_cast<const char*>(&token_length), sizeof(token_length));
        os.write(token.c_str(), token_length);
        os.write(reinterpret_cast<const char*>(&id), sizeof(id));
    }
}

std::vector<std::string> TiktokenTokenizer::get_vocabulary_vector() const {
    if (!is_initialized()) throw std::runtime_error("Tokenizer not initialized");
    
    std::vector<std::string> vocab;
    vocab.reserve(pimpl_->vocab_size());
    
    // Get all tokens by decoding their IDs
    for (size_t i = 0; i < pimpl_->vocab_size(); i++) {
        vocab.push_back(decode({static_cast<int>(i)}));
    }
    
    return vocab;
}

void TiktokenTokenizer::initialize_token_categories() {
    // Initialize verb tokens
    verb_tokens_ = {
        "run", "walk", "jump", "eat", "sleep", "write", "read",
        "speak", "listen", "think", "feel", "see", "hear", "touch",
        "smell", "taste", "move", "stop", "start", "finish"
    };

    // Initialize adjective tokens
    adjective_tokens_ = {
        "big", "small", "fast", "slow", "hot", "cold", "good", "bad",
        "happy", "sad", "angry", "calm", "loud", "quiet", "bright", "dark"
    };

    // Initialize noun tokens
    noun_tokens_ = {
        "man", "woman", "child", "dog", "cat", "house", "car", "tree",
        "book", "phone", "computer", "food", "water", "time", "day", "night"
    };

    // Initialize determiner tokens
    determiner_tokens_ = {
        "the", "a", "an", "this", "that", "these", "those", "my",
        "your", "his", "her", "its", "our", "their", "any", "some"
    };
}

// Helper to identify if a phrase is a verb or adjective
bool is_phrase_type(const std::string& phrase, bool check_verbs = false) {
    // Check for marked patterns
    if (check_verbs) {
        if (phrase.find('#') != std::string::npos) {
            return true;
        }
    } else {
        if (phrase.find('*') != std::string::npos) {
            return true;
        }
    }
    
    // Common endings for the respective type
    static const std::unordered_set<std::string> adjective_endings = {
        "able", "ible", "al", "ful", "ic", "ive", "less", "ous", "y"
    };
    
    static const std::unordered_set<std::string> verb_endings = {
        "ate", "ize", "ify", "ise", "ect", "ent", "age", "ute", "ing", "ed", "es", "s"
    };
    
    const auto& endings = check_verbs ? verb_endings : adjective_endings;
    
    // Common words of the respective type
    static const std::unordered_set<std::string> common_adjectives = {
        " bright", " dark", " hot", " cold", " big", " small", " tall", " short",
        " red", " blue", " green", " black", " white", " yellow", " good", " bad",
        " fast", " slow", " hard", " soft", " loud", " quiet", " rich", " poor",
        " young", " old", " new", " old", " happy", " sad", " clean", " dirty"
    };
    
    static const std::unordered_set<std::string> common_verbs = {
        " organize", " manage", " coordinate", " direct", " lead", " guide", " plan",
        " create", " build", " make", " write", " speak", " teach", " learn",
        " analyze", " study", " work", " play", " help", " start", " finish"
    };
    
    const auto& common_words = check_verbs ? common_verbs : common_adjectives;
    
    // First check if it's a common word
    if (common_words.find(phrase) != common_words.end()) {
        return true;
    }
    
    // Then check for endings
    for (const auto& ending : endings) {
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

        // For marked adjective patterns (with *)
        size_t asterisk_pos = line.find('*');
        if (asterisk_pos != std::string::npos) {
            std::string context = line.substr(0, asterisk_pos);
            std::string target = line.substr(asterisk_pos); // Keep the * in the target
            
            // Trim context
            context.erase(0, context.find_first_not_of(" \t\r\n"));
            context.erase(context.find_last_not_of(" \t\r\n") + 1);
            
            // Add a space prefix to target if needed
            if (!target.empty() && target[0] != ' ' && target[0] != '*') {
                target = " " + target;
            }
            
            if (!context.empty() && !target.empty()) {
                pairs.emplace_back(context, target);
            }
            continue;
        }

        // For marked verb patterns (with #)
        size_t hash_pos = line.find('#');
        if (hash_pos != std::string::npos) {
            std::string context = line.substr(0, hash_pos);
            std::string target = line.substr(hash_pos); // Keep the # in the target
            
            // Trim context
            context.erase(0, context.find_first_not_of(" \t\r\n"));
            context.erase(context.find_last_not_of(" \t\r\n") + 1);
            
            // Add a space prefix to target if needed
            if (!target.empty() && target[0] != ' ' && target[0] != '#') {
                target = " " + target;
            }
            
            if (!context.empty() && !target.empty()) {
                pairs.emplace_back(context, target);
            }
            continue;
        }

        // For regular patterns (with |)
        size_t sep_pos = line.find('|');
        if (sep_pos != std::string::npos) {
            std::string context = line.substr(0, sep_pos);
            std::string target = line.substr(sep_pos + 1);
            
            context.erase(0, context.find_first_not_of(" \t\r\n"));
            context.erase(context.find_last_not_of(" \t\r\n") + 1);
            
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
    if (!initialized_) {
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
    if (!initialized_) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    
    auto it = id_to_token_.find(token_id);
    if (it != id_to_token_.end()) {
        return it->second;
    }
    
    return "<unk>";  // Return unknown token string if not found
} 