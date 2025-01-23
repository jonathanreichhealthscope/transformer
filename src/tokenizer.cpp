#include "../include/tokenizer.hpp"
#include <algorithm>
#include <regex>
#include <sstream>
#include <stdexcept>
#include "../include/cuda/matrix_ops.cuh"
#include "../include/cuda/tokenizer_kernels.cuh"

// Define the special character map
const std::unordered_map<char, std::string> Tokenizer::SPECIAL_CHAR_MAP = {
    {'\n', "<newline>"},    {'\t', "<tab>"},     {'.', "<period>"},
    {'!', "<exclamation>"}, {'?', "<question>"}, {',', "<comma>"}};

Tokenizer::Tokenizer() : vocab(std::make_unique<Vocabulary>()) {}

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

    try {
#ifdef USE_CUDA
        try {
            // Use CUDA for parallel token matching
            std::vector<int> tokens;
            cuda::parallel_tokenize(text, *vocab, tokens);
            encoding_cache[text] = tokens;
            return tokens;
        } catch (const std::runtime_error& e) {
            std::cerr << "CUDA tokenization failed, falling back to CPU: " << e.what() << std::endl;
#endif
            // CPU fallback implementation
            std::vector<int> tokens;
            size_t pos = 0;
            while (pos < text.length()) {
                // Find longest matching token at current position
                int longest_token = vocab->get_unk_token_id();
                size_t longest_len = 0;
                
                for (size_t i = 0; i < vocab->size(); i++) {
                    std::string token = vocab->get_token(i);
                    if (text.compare(pos, token.length(), token) == 0) {
                        if (token.length() > longest_len) {
                            longest_token = i;
                            longest_len = token.length();
                        }
                    }
                }

                tokens.push_back(longest_token);
                pos += longest_len > 0 ? longest_len : 1;
            }
            encoding_cache[text] = tokens;
            return tokens;
#ifdef USE_CUDA
        }
#endif
    } catch (const std::exception& e) {
        throw std::runtime_error("Tokenization failed: " + std::string(e.what()));
    }
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    bool skip_space = false; // To handle consecutive whitespace tokens

    for (int token : tokens) {
        std::string token_str = vocab->get_token(token);

        // Skip special control tokens
        if (token_str == "<pad>" || token_str == "<bos>" || token_str == "<eos>") {
            continue;
        }

        // Handle special character tokens
        if (token_str == "<whitespace>") {
            if (!skip_space) {
                result += " ";
                skip_space = true;
            }
        } else if (token_str == "<newline>") {
            result += "\n";
            skip_space = false;
        } else if (token_str == "<tab>") {
            result += "\t";
            skip_space = false;
        } else if (token_str == "<period>") {
            result += ".";
            skip_space = false;
        } else if (token_str == "<exclamation>") {
            result += "!";
            skip_space = false;
        } else if (token_str == "<question>") {
            result += "?";
            skip_space = false;
        } else if (token_str == "<comma>") {
            result += ",";
            skip_space = false;
        } else {
            result += token_str;
            skip_space = false;
        }
    }

    return result;
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