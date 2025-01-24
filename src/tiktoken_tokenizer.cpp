#include "../include/tiktoken_tokenizer.hpp"
#include <stdexcept>
#include <iostream>
#include <sstream>

TiktokenTokenizer::TiktokenTokenizer() {
    // Default initialization with cl100k_base encoding
    initialize();
}

void TiktokenTokenizer::initialize(const std::string& encoding_name) {
    try {
        std::cout << "Initializing tiktoken with encoding: " << encoding_name << std::endl;
        tiktoken_ = std::make_unique<tiktoken::Encoding>(encoding_name);
        std::cout << "Base vocabulary size before special tokens: " << tiktoken_->get_vocab_size() << std::endl;
        setup_special_tokens();
        std::cout << "Final vocabulary size after special tokens: " << tiktoken_->get_vocab_size() << std::endl;
        
        // Test encode a simple string
        std::string test_str = "Hello world";
        auto test_tokens = tiktoken_->encode(test_str);
        std::cout << "Test encoding '" << test_str << "': ";
        for (auto token : test_tokens) {
            std::cout << token << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize tiktoken: " + std::string(e.what()));
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

std::vector<int> TiktokenTokenizer::encode(const std::string& text) const {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }

    try {
        if (text.empty()) {
            std::cout << "Warning: Attempting to encode empty string" << std::endl;
            return std::vector<int>();
        }

        // Simple whitespace tokenization as temporary solution
        std::vector<int> tokens;
        std::string word;
        std::istringstream iss(text);
        
        // Add BOS token if not empty
        tokens.push_back(tokens::BOS_ID);
        
        // Split on whitespace and assign temporary token IDs
        while (iss >> word) {
            // For now, just hash the word to get a token ID, ensuring it's within vocab range
            size_t hash_val = std::hash<std::string>{}(word);
            int token_id = (hash_val % (tiktoken_->get_vocab_size() - tokens::NUM_SPECIAL_TOKENS)) + tokens::NUM_SPECIAL_TOKENS;
            tokens.push_back(token_id);
        }
        
        // Add EOS token
        tokens.push_back(tokens::EOS_ID);

        if (tokens.empty()) {
            std::cout << "Warning: Encoding produced empty tokens for text: '" << text << "'" << std::endl;
        } else {
            std::cout << "Encoded '" << text << "' to " << tokens.size() << " tokens" << std::endl;
        }
        return tokens;
    } catch (const std::exception& e) {
        std::cout << "Error encoding text: '" << text << "': " << e.what() << std::endl;
        throw std::runtime_error("Encoding failed: " + std::string(e.what()));
    }
}

std::string TiktokenTokenizer::decode(const std::vector<int>& tokens) const {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }

    try {
        std::string result;
        for (size_t i = 0; i < tokens.size(); i++) {
            int token = tokens[i];
            // Skip special tokens
            if (token == tokens::BOS_ID || token == tokens::EOS_ID) {
                continue;
            }
            // For now, just return the token ID as a string
            if (!result.empty()) {
                result += " ";
            }
            result += "<" + std::to_string(token) + ">";
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
    // Special tokens are already included in tiktoken's vocabulary
    return tiktoken_->get_vocab_size();
} 