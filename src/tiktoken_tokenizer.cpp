#include "../include/tiktoken_tokenizer.hpp"
#include <stdexcept>
#include <iostream>
#include <sstream>

TiktokenTokenizer::TiktokenTokenizer() = default;

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

        // Use tiktoken's encode method directly instead of custom tokenization
        auto tokens = tiktoken_->encode(text);
        
        // Add BOS and EOS tokens
        std::vector<int> result;
        result.reserve(tokens.size() + 2);
        result.push_back(tokens::BOS_ID);
        result.insert(result.end(), tokens.begin(), tokens.end());
        result.push_back(tokens::EOS_ID);

        if (result.empty()) {
            std::cout << "Warning: Encoding produced empty tokens for text: '" << text << "'" << std::endl;
        } 
        return result;
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
        // Filter out special tokens
        std::vector<int> filtered_tokens;
        filtered_tokens.reserve(tokens.size());
        for (int token : tokens) {
            if (token != tokens::BOS_ID && token != tokens::EOS_ID) {
                filtered_tokens.push_back(token);
            }
        }
        
        // Use tiktoken's decode method
        return tiktoken_->decode(filtered_tokens);
    } catch (const std::exception& e) {
        throw std::runtime_error("Decoding failed: " + std::string(e.what()));
    }
}

size_t TiktokenTokenizer::vocab_size() const {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    return tiktoken_->get_vocab_size();
} 