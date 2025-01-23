#include "../include/tokenizer.hpp"
#include <algorithm>
#include <regex>
#include <sstream>
#include <stdexcept>
#include "../include/cuda/matrix_ops.cuh"
#include "../include/cuda/tokenizer_kernels.cuh"
#include "../include/sentencepiece_tokenizer.hpp"

// Define the special character map
const std::unordered_map<char, std::string> Tokenizer::SPECIAL_CHAR_MAP = {
    {'\n', "<newline>"},    {'\t', "<tab>"},     {'.', "<period>"},
    {'!', "<exclamation>"}, {'?', "<question>"}, {',', "<comma>"}};

Tokenizer::Tokenizer() 
    : vocab(std::make_unique<Vocabulary>())
    , tokenizer_(std::make_unique<SentencePieceTokenizer>()) {
    
    // Initialize vocabulary with special tokens
    vocab->add_special_token(tokens::PAD_TOKEN, tokens::PAD_ID);
    vocab->add_special_token(tokens::UNK_TOKEN, tokens::UNK_ID);
    vocab->add_special_token(tokens::BOS_TOKEN, tokens::BOS_ID);
    vocab->add_special_token(tokens::EOS_TOKEN, tokens::EOS_ID);
    vocab->add_special_token(tokens::MASK_TOKEN, tokens::MASK_ID);
}

void Tokenizer::save_vocabulary(std::ostream& os) const {
    size_t vocab_size = vocab->size();
    os.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));

    for (size_t i = 0; i < vocab_size; ++i) {
        std::string token = vocab->get_token(i);
        size_t token_length = token.length();
        os.write(reinterpret_cast<const char*>(&token_length), sizeof(token_length));
        os.write(token.c_str(), token_length);
    }
}

std::unique_ptr<Vocabulary> Tokenizer::load_vocabulary(std::istream& is) {
    auto vocab = std::make_unique<Vocabulary>();

    size_t vocab_size;
    is.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));

    for (size_t i = 0; i < vocab_size; ++i) {
        size_t token_length;
        is.read(reinterpret_cast<char*>(&token_length), sizeof(token_length));

        std::vector<char> token_buffer(token_length + 1, '\0');
        is.read(token_buffer.data(), token_length);
        std::string token(token_buffer.data());

        vocab->add_special_token(token, i);
    }

    return vocab;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    // Check cache first
    auto cache_it = encoding_cache.find(text);
    if (cache_it != encoding_cache.end()) {
        return cache_it->second;
    }

    auto ids = tokenizer_->encode(text);
    encoding_cache[text] = ids;
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    return tokenizer_->decode(tokens);
}

void Tokenizer::save(std::ostream& os) const {
    try {
        uint32_t version = 1;
        os.write(reinterpret_cast<const char*>(&version), sizeof(version));
        save_vocabulary(os);
        if (!os.good()) {
            throw std::runtime_error("Failed to save tokenizer");
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Error saving tokenizer: " + std::string(e.what()));
    }
}

std::unique_ptr<Tokenizer> Tokenizer::load(std::istream& is) {
    try {
        auto tokenizer = std::make_unique<Tokenizer>();

        uint32_t version;
        is.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) {
            throw std::runtime_error("Unsupported tokenizer version");
        }

        tokenizer->vocab = load_vocabulary(is);

        if (!is.good()) {
            throw std::runtime_error("Failed to load tokenizer");
        }

        return tokenizer;
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading tokenizer: " + std::string(e.what()));
    }
}

bool Tokenizer::is_special_token(int token_id) const {
    std::string token = vocab->get_token(token_id);
    return token == "<pad>" || token == "<unk>" || token == "<bos>" || token == "<eos>" ||
           token == "<whitespace>" || token.find("<") == 0; // Check for other special tokens
}

void Tokenizer::preprocess_text(std::string& text) const {
    std::string result;
    bool in_whitespace = false;

    for (size_t i = 0; i < text.length(); ++i) {
        char c = text[i];

        // Handle whitespace
        if (std::isspace(c)) {
            if (!in_whitespace) {
                result += " <whitespace> ";
                in_whitespace = true;
            }
            continue;
        }
        in_whitespace = false;

        // Handle special characters
        auto it = SPECIAL_CHAR_MAP.find(c);
        if (it != SPECIAL_CHAR_MAP.end()) {
            result += it->second;
        } else {
            result += c;
        }
    }

    text = result;
}

void Tokenizer::sync_vocabulary_with_subword_tokenizer() {
    if (!tokenizer_) return;
    
    // Create new vocabulary preserving special tokens
    auto new_vocab = std::make_unique<Vocabulary>();
    
    // First add special tokens in correct order
    new_vocab->add_special_token(tokens::PAD_TOKEN, tokens::PAD_ID);
    new_vocab->add_special_token(tokens::UNK_TOKEN, tokens::UNK_ID);
    new_vocab->add_special_token(tokens::BOS_TOKEN, tokens::BOS_ID);
    new_vocab->add_special_token(tokens::EOS_TOKEN, tokens::EOS_ID);
    new_vocab->add_special_token(tokens::MASK_TOKEN, tokens::MASK_ID);
    
    // Then add all other tokens from SentencePiece tokenizer
    for (size_t i = tokens::NUM_SPECIAL_TOKENS; i < tokenizer_->vocab_size(); i++) {
        std::string token = tokenizer_->id_to_token(i);
        new_vocab->add_token(token, i);
    }
    
    // Verify consistency before replacing
    if (!new_vocab->verify_mappings()) {
        throw std::runtime_error("Inconsistent mappings after vocabulary sync");
    }
    
    vocab = std::move(new_vocab);
}