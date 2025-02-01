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
    size_t vocab_size_ = 2857;  // This will be properly set by set_vocab_size
    std::unordered_map<std::string, int> original_vocab;

    void initialize(const std::string& encoding_name) {
        if (initialized_) return;
        
        // Initialize with special tokens first
        vocab[tokens::PAD_TOKEN] = tokens::PAD_ID;
        vocab[tokens::UNK_TOKEN] = tokens::UNK_ID;
        vocab[tokens::BOS_TOKEN] = tokens::BOS_ID;
        vocab[tokens::EOS_TOKEN] = tokens::EOS_ID;
        vocab[tokens::MASK_TOKEN] = tokens::MASK_ID;
        vocab[tokens::SEP_TOKEN] = tokens::SEP_ID;
        
        // Store initial vocabulary size
        vocab_size_ = tokens::NUM_SPECIAL_TOKENS;
        
        // Store original vocab for potential resizing
        original_vocab = vocab;
        
        initialized_ = true;
        std::cout << "Initialized tokenizer with " << vocab_size_ 
                  << " tokens (special tokens only)" << std::endl;
    }

    bool is_initialized() const { return initialized_; }

    std::vector<int> encode(const std::string& text) const {
        if (!initialized_) throw std::runtime_error("Tokenizer not initialized");
        
        std::vector<int> tokens;
        std::istringstream iss(text);
        std::string word;
        
        // Split text into words
        while (iss >> word) {
            // Check if the word exists in our vocabulary
            auto it = vocab.find(word);
            if (it != vocab.end() && static_cast<size_t>(it->second) < vocab_size_) {
                tokens.push_back(it->second);
            } else {
                // If word not found or ID is outside current vocab size, use UNK token
                tokens.push_back(tokens::UNK_ID);
                std::cout << "Unknown token or out of vocab range: '" << word 
                          << "' (vocab_size: " << vocab_size_ << ")" << std::endl;
            }
        }
        
        if (tokens.empty()) {
            std::cout << "Warning: Empty tokens generated for text: '" << text << "'" << std::endl;
            return {tokens::UNK_ID};  // Return UNK token for empty input
        }
        
        return tokens;
    }

    std::string decode(const std::vector<int>& tokens) const {
        if (!initialized_) throw std::runtime_error("Tokenizer not initialized");
        
        std::string result;
        bool first = true;
        
        for (int token_id : tokens) {
            // Skip if token ID is outside our current vocabulary size
            if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size_) {
                std::cout << "Warning: Token ID " << token_id 
                          << " is outside vocabulary range (0-" << vocab_size_ - 1 
                          << ")" << std::endl;
                continue;
            }
            
            // Find the token string for this ID
            std::string token_str;
            bool found = false;
            for (const auto& [str, id] : vocab) {
                if (id == token_id) {
                    token_str = str;
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                std::cout << "Warning: No string found for token ID " << token_id << std::endl;
                token_str = tokens::UNK_TOKEN;
            }
            
            // Add space between tokens (except for first token and special tokens)
            if (!first && !token_str.empty() && 
                token_id != tokens::PAD_ID && 
                token_id != tokens::UNK_ID && 
                token_id != tokens::BOS_ID && 
                token_id != tokens::EOS_ID && 
                token_id != tokens::MASK_ID && 
                token_id != tokens::SEP_ID) {
                result += " ";
            }
            
            result += token_str;
            first = false;
        }
        
        return result;
    }

    size_t vocab_size() const {
        if (!initialized_) throw std::runtime_error("Tokenizer not initialized");
        return vocab_size_;
    }

    void resize_vocabulary(size_t new_size) {
        if (!initialized_) {
            throw std::runtime_error("Cannot resize vocabulary before initialization");
        }
        
        std::cout << "Resizing vocabulary from " << vocab_size_ << " to " << new_size << std::endl;
        
        // Ensure we don't go below the number of special tokens
        const size_t min_size = tokens::NUM_SPECIAL_TOKENS;
        if (new_size < min_size) {
            std::cout << "Warning: Requested vocab size " << new_size 
                      << " is below minimum size " << min_size 
                      << ". Using minimum size." << std::endl;
            new_size = min_size;
        }
        
        // If we haven't stored the original vocab yet, do so
        if (original_vocab.empty()) {
            original_vocab = vocab;
            std::cout << "Stored original vocabulary with " << original_vocab.size() << " tokens" << std::endl;
        }
        
        // Clear current vocab
        vocab.clear();
        
        // First, add all special tokens
        for (int i = 0; i < tokens::NUM_SPECIAL_TOKENS; i++) {
            std::string special_token = get_special_token_string(i);
            auto it = original_vocab.find(special_token);
            if (it != original_vocab.end()) {
                vocab[it->first] = it->second;
                std::cout << "Added special token: " << it->first << " with ID: " << it->second << std::endl;
            }
        }
        
        // Then add remaining tokens up to new_size
        size_t current_size = vocab.size();
        for (const auto& [token, id] : original_vocab) {
            if (current_size >= new_size) break;
            
            // Skip if it's already in vocab (special tokens)
            if (vocab.find(token) == vocab.end()) {
                vocab[token] = id;
                current_size++;
            }
        }
        
        vocab_size_ = new_size;
        std::cout << "Vocabulary resized. New size: " << vocab_size_ 
                  << ", Active tokens: " << vocab.size() << std::endl;
    }
    
    std::string get_special_token_string(int index) const {
        switch(index) {
            case 0: return tokens::PAD_TOKEN;
            case 1: return tokens::UNK_TOKEN;
            case 2: return tokens::BOS_TOKEN;
            case 3: return tokens::EOS_TOKEN;
            case 4: return tokens::MASK_TOKEN;
            case 5: return tokens::SEP_TOKEN;
            default: return "";
        }
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
    if (!pimpl_ || !pimpl_->initialized_) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    
    // Use the truncated vocabulary for encoding
    std::vector<int> tokens;
    std::string current_token;
    std::istringstream iss(text);
    
    while (iss >> current_token) {
        auto it = pimpl_->vocab.find(current_token);
        if (it != pimpl_->vocab.end()) {
            tokens.push_back(it->second);
        } else {
            tokens.push_back(get_unk_token_id());
        }
    }
    
    return tokens;
}

std::string TiktokenTokenizer::decode(const std::vector<int>& tokens) const {
    if (!pimpl_ || !pimpl_->initialized_) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    
    std::string result;
    for (int token : tokens) {
        // Find the token string in our truncated vocabulary
        bool found = false;
        for (const auto& [str, id] : pimpl_->vocab) {
            if (id == token) {
                result += (result.empty() ? "" : " ") + str;
                found = true;
                break;
            }
        }
        if (!found) {
            result += (result.empty() ? "" : " ") + tokens::UNK_TOKEN;
        }
    }
    
    return result;
}

size_t TiktokenTokenizer::vocab_size() const {
    if (!pimpl_) {
        throw std::runtime_error("Tokenizer implementation not initialized");
    }
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
    
    return "";
}

// Implement the public set_vocab_size method
void TiktokenTokenizer::set_vocab_size(size_t size) {
    if (!pimpl_) {
        throw std::runtime_error("Tokenizer implementation not initialized");
    }
    
    std::cout << "Setting vocabulary size to " << size << std::endl;
    
    try {
        // Resize the vocabulary
        pimpl_->resize_vocabulary(size);
        
        // Verify the change
        size_t new_size = pimpl_->vocab_size();
        if (new_size != size && new_size != tokens::NUM_SPECIAL_TOKENS) {
            throw std::runtime_error("Failed to set vocabulary size. Expected: " + 
                                   std::to_string(size) + ", Got: " + 
                                   std::to_string(new_size));
        }
        
        std::cout << "Successfully set vocabulary size to " << new_size << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error setting vocabulary size: " << e.what() << std::endl;
        throw;
    }
}

// Add a method to verify the vocabulary size
void TiktokenTokenizer::print_vocabulary_stats() const {
    if (!pimpl_) {
        throw std::runtime_error("Tokenizer implementation not initialized");
    }
    
    std::cout << "\nVocabulary Statistics:" << std::endl;
    std::cout << "- Total vocabulary size: " << pimpl_->vocab_size_ << std::endl;
    std::cout << "- Active tokens: " << pimpl_->vocab.size() << std::endl;
    std::cout << "- Special tokens: " << tokens::NUM_SPECIAL_TOKENS << std::endl;
    
    // Print first few tokens for verification
    std::cout << "\nFirst few tokens:" << std::endl;
    int count = 0;
    for (const auto& [token, id] : pimpl_->vocab) {
        if (count++ < 10) {
            std::cout << "  " << token << " -> " << id << std::endl;
        } else {
            break;
        }
    }
}