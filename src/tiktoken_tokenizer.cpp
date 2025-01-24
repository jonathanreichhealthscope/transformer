#include "../include/tiktoken_tokenizer.hpp"
#include <stdexcept>
#include <iostream>

TiktokenTokenizer::TiktokenTokenizer() {
    // Default initialization with cl100k_base encoding
    initialize();
}

void TiktokenTokenizer::initialize(const std::string& encoding_name) {
    try {
        tiktoken_ = std::make_unique<tiktoken::Encoding>(encoding_name);
        setup_special_tokens();
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize tiktoken: " + std::string(e.what()));
    }
}

void TiktokenTokenizer::setup_special_tokens() {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }

    // Add our special tokens to tiktoken's vocabulary
    tiktoken_->add_special_token("<pad>", tokens::PAD_ID);
    tiktoken_->add_special_token("<unk>", tokens::UNK_ID);
    tiktoken_->add_special_token("<s>", tokens::BOS_ID);
    tiktoken_->add_special_token("</s>", tokens::EOS_ID);
    tiktoken_->add_special_token("<mask>", tokens::MASK_ID);
}

std::vector<int> TiktokenTokenizer::encode(const std::string& text) const {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }

    try {
        return tiktoken_->encode(text);
    } catch (const std::exception& e) {
        throw std::runtime_error("Encoding failed: " + std::string(e.what()));
    }
}

std::string TiktokenTokenizer::decode(const std::vector<int>& tokens) const {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }

    try {
        return tiktoken_->decode(tokens);
    } catch (const std::exception& e) {
        throw std::runtime_error("Decoding failed: " + std::string(e.what()));
    }
}

size_t TiktokenTokenizer::vocab_size() const {
    if (!is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    // Add our special tokens to the base vocabulary size
    return tiktoken_->get_vocab_size() + tokens::NUM_SPECIAL_TOKENS;
} 