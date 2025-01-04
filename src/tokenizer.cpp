#include "../include/tokenizer.hpp"
#include <stdexcept>
#include <sstream>
#include <regex>
#include <algorithm>

// Define the special character map
extern const std::unordered_map<char, std::string> SPECIAL_CHAR_MAP;

Tokenizer::Tokenizer() : vocab(std::make_unique<Vocabulary>()) {}

void Tokenizer::save_vocabulary(std::ostream &os) const {
    size_t vocab_size = vocab->size();
    os.write(reinterpret_cast<const char *>(&vocab_size), sizeof(vocab_size));

    for (size_t i = 0; i < vocab_size; ++i) {
        std::string token = vocab->get_token(i);
        size_t token_length = token.length();
        os.write(reinterpret_cast<const char *>(&token_length), sizeof(token_length));
        os.write(token.c_str(), token_length);
    }
}

std::unique_ptr<Vocabulary> Tokenizer::load_vocabulary(std::istream &is) {
    auto vocab = std::make_unique<Vocabulary>();

    size_t vocab_size;
    is.read(reinterpret_cast<char *>(&vocab_size), sizeof(vocab_size));

    for (size_t i = 0; i < vocab_size; ++i) {
        size_t token_length;
        is.read(reinterpret_cast<char *>(&token_length), sizeof(token_length));

        std::vector<char> token_buffer(token_length + 1, '\0');
        is.read(token_buffer.data(), token_length);
        std::string token(token_buffer.data());

        vocab->add_special_token(token, i);
    }

    return vocab;
}

std::vector<int> Tokenizer::encode(const std::string &text) const {
    // Check cache first
    auto cache_it = encoding_cache.find(text);
    if (cache_it != encoding_cache.end()) {
        return cache_it->second;
    }
    
    std::vector<int> tokens;
    tokens.push_back(vocab->get_id("<bos>"));
    
    // Use subword tokenization instead of word-level
    std::string current_subword;
    for (size_t i = 0; i < text.length(); i++) {
        current_subword += text[i];
        if (vocab->has_token(current_subword)) {
            tokens.push_back(vocab->get_id(current_subword));
            current_subword.clear();
        } else if (current_subword.length() > MAX_SUBWORD_LENGTH) {
            // Fallback to character-level if subword not found
            tokens.push_back(vocab->get_id(std::string(1, text[i-1])));
            current_subword = text[i];
        }
    }
    
    // Handle any remaining subword
    if (!current_subword.empty()) {
        tokens.push_back(vocab->get_id("<unk>"));
    }
    
    tokens.push_back(vocab->get_id("<eos>"));
    
    // Cache result before returning
    encoding_cache[text] = tokens;
    return tokens;
}

std::string Tokenizer::decode(const std::vector<int> &tokens) const {
    std::string result;
    for (int token : tokens) {
        std::string token_str = vocab->get_token(token);
        if (token_str != "<pad>" && token_str != "<bos>" && token_str != "<eos>") {
            // Handle special tokens during decoding
            if (token_str == "<whitespace>") {
                result += " ";
            } else if (token_str == "<newline>") {
                result += "\n";
            } else if (token_str == "<tab>") {
                result += "\t";
            } else {
                result += token_str;
            }
        }
    }
    return result;
}

void Tokenizer::save(std::ostream &os) const {
    try {
        uint32_t version = 1;
        os.write(reinterpret_cast<const char *>(&version), sizeof(version));
        save_vocabulary(os);
        if (!os.good()) {
            throw std::runtime_error("Failed to save tokenizer");
        }
    } catch (const std::exception &e) {
        throw std::runtime_error("Error saving tokenizer: " + std::string(e.what()));
    }
}

std::unique_ptr<Tokenizer> Tokenizer::load(std::istream &is) {
    try {
        auto tokenizer = std::make_unique<Tokenizer>();

        uint32_t version;
        is.read(reinterpret_cast<char *>(&version), sizeof(version));
        if (version != 1) {
            throw std::runtime_error("Unsupported tokenizer version");
        }

        tokenizer->vocab = load_vocabulary(is);

        if (!is.good()) {
            throw std::runtime_error("Failed to load tokenizer");
        }

        return tokenizer;
    } catch (const std::exception &e) {
        throw std::runtime_error("Error loading tokenizer: " + std::string(e.what()));
    }
}

bool Tokenizer::is_special_token(int token_id) const {
    std::string token = vocab->get_token(token_id);
    return token == "<pad>" || token == "<unk>" || token == "<bos>" ||
           token == "<eos>" || token == "<whitespace>" || 
           token.find("<") == 0; // Check for other special tokens
}

void Tokenizer::preprocess_text(std::string &text) const {
    // Preserve whitespace with special tokens
    std::regex whitespace_re("\\s+");
    text = std::regex_replace(text, whitespace_re, " <whitespace> ");
    
    // Handle special characters
    for (const auto& [special_char, token] : SPECIAL_CHAR_MAP) {
        std::string char_str(1, special_char);
        text = std::regex_replace(text, std::regex(std::regex_replace(char_str, std::regex("[.?*+^$()\\[\\]{}|\\\\]"), "\\$&")), token);
    }
}