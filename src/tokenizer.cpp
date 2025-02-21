#include "../include/tokenizer.hpp"
#include "../include/tiktoken_tokenizer.hpp"
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <regex>

// Define the special character map
const std::unordered_map<char, std::string> Tokenizer::SPECIAL_CHAR_MAP = {
    {'\n', "<newline>"},    {'\t', "<tab>"},     {'.', "<period>"},
    {'!', "<exclamation>"}, {'?', "<question>"}, {',', "<comma>"}};

std::vector<int> Tokenizer::encode(const std::string& text) const {
    return tokenizer_->encode(text);
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    return tokenizer_->decode(tokens);
}

void Tokenizer::preprocess_text(std::string& text) const {
    std::string result;
    bool in_whitespace = false;

    for (size_t i = 0; i < text.length(); ++i) {
        char c = text[i];
        if (std::isspace(c)) {
            if (!in_whitespace) {
                result += " ";
                in_whitespace = true;
            }
            continue;
        }
        in_whitespace = false;

        auto it = SPECIAL_CHAR_MAP.find(c);
        if (it != SPECIAL_CHAR_MAP.end()) {
            result += it->second;
        } else {
            result += c;
        }
    }
    text = result;
}

bool Tokenizer::is_special_token(int token_id) const {
    std::string token = tokenizer_->decode({token_id});
    return token.find("<") == 0 && token.find(">") == token.length() - 1;
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
        uint32_t version;
        is.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) {
            throw std::runtime_error("Unsupported tokenizer version");
        }

        // Read vocabulary size
        uint32_t vocab_size;
        is.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));

        // Read special token IDs
        int32_t special_tokens[5];
        is.read(reinterpret_cast<char*>(special_tokens), sizeof(special_tokens));

        // Create and return a new tokenizer (specific type should be determined by the file format)
        // For now, we'll throw an error since we need to implement proper deserialization
        throw std::runtime_error("Tokenizer loading not yet implemented");

    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading tokenizer: " + std::string(e.what()));
    }
}

void Tokenizer::save_vocabulary(std::ostream& os) const {
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not initialized");
    }

    // Write vocabulary size
    uint32_t vocab_size = static_cast<uint32_t>(tokenizer_->vocab_size());
    os.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));

    // Write special token IDs
    int32_t special_tokens[] = {
        tokenizer_->get_pad_token_id(),
        tokenizer_->get_unk_token_id(),
        tokenizer_->get_bos_token_id(),
        tokenizer_->get_eos_token_id(),
        tokenizer_->get_mask_token_id()
    };
    os.write(reinterpret_cast<const char*>(special_tokens), sizeof(special_tokens));

    // Get vocabulary
    auto vocab = get_vocabulary_vector();
    
    // Write each token
    for (const auto& token : vocab) {
        uint32_t token_length = static_cast<uint32_t>(token.length());
        os.write(reinterpret_cast<const char*>(&token_length), sizeof(token_length));
        os.write(token.c_str(), token_length);
    }
}

std::unique_ptr<TiktokenTokenizer> Tokenizer::load_vocabulary(std::istream& is) {
    auto tokenizer = std::make_unique<TiktokenTokenizer>();

    // Initialize with encoding name
    tokenizer->initialize("cl100k_base");

    // Read vocabulary size (for compatibility with file format)
    uint32_t vocab_size;
    is.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));

    // Read special token IDs (for compatibility with file format)
    int32_t special_tokens[5];
    is.read(reinterpret_cast<char*>(special_tokens), sizeof(special_tokens));

    // Skip token data since we're using the pre-defined vocabulary
    for (uint32_t i = 0; i < vocab_size; ++i) {
        uint32_t token_length;
        is.read(reinterpret_cast<char*>(&token_length), sizeof(token_length));
        is.seekg(token_length, std::ios::cur);  // Skip token data
    }

    return tokenizer;
}

void Tokenizer::print_vocabulary_mappings() const {
    if (!tokenizer_ || !tokenizer_->is_initialized()) {
        std::cerr << "Warning: Attempting to print mappings before tokenizer initialization" << std::endl;
        return;
    }

    std::cout << "Special Token IDs:\n"
              << "PAD: " << tokenizer_->get_pad_token_id() << "\n"
              << "UNK: " << tokenizer_->get_unk_token_id() << "\n"
              << "BOS: " << tokenizer_->get_bos_token_id() << "\n"
              << "EOS: " << tokenizer_->get_eos_token_id() << "\n"
              << "MASK: " << tokenizer_->get_mask_token_id() << std::endl;
}

std::vector<std::string> Tokenizer::get_vocabulary_vector() const {
    if (!tokenizer_ || !tokenizer_->is_initialized()) {
        throw std::runtime_error("Tokenizer not initialized");
    }

    std::vector<std::string> vocab;
    const size_t vocab_size = tokenizer_->vocab_size();
    vocab.reserve(vocab_size);
    
    // Get all tokens by decoding their IDs
    for (size_t i = 0; i < vocab_size; i++) {
        vocab.push_back(tokenizer_->decode({static_cast<int>(i)}));
    }
    
    return vocab;
}

bool Tokenizer::is_noun(const std::string& token) const {
    // Simple heuristic: consider capitalized words or special tokens as nouns
    return !token.empty() && (
        std::isupper(token[0]) ||
        token.find("<") == 0  // Special tokens
    );
}

bool Tokenizer::is_adjective(const std::string& token) const {
    // Common adjectives
    static const std::unordered_set<std::string> adjectives = {
        "big", "small", "large", "tiny", "huge", "little",
        "good", "bad", "great", "awful", "terrible", "wonderful",
        "beautiful", "ugly", "pretty", "handsome",
        "old", "new", "young", "ancient", "modern",
        "happy", "sad", "angry", "excited", "nervous",
        "red", "blue", "green", "yellow", "black", "white",
        "hot", "cold", "warm", "cool",
        "fast", "slow", "quick", "rapid",
        "hard", "soft", "rough", "smooth",
        "bright", "dark", "dim", "shiny",
        "loud", "quiet", "noisy", "silent",
        "clean", "dirty", "neat", "messy",
        "rich", "poor", "wealthy", "expensive",
        "strong", "weak", "powerful", "feeble",
        "smart", "clever", "intelligent", "wise",
        "brave", "cowardly", "fearless", "timid",
        "kind", "mean", "gentle", "cruel",
        "tall", "short", "high", "low",
        "wide", "narrow", "broad", "thin",
        "deep", "shallow", "thick", "slim"
    };
    
    // Convert token to lowercase for comparison
    std::string lower_token = token;
    std::transform(lower_token.begin(), lower_token.end(), lower_token.begin(), ::tolower);
    
    return adjectives.find(lower_token) != adjectives.end();
}

bool Tokenizer::is_determiner(const std::string& token) const {
    // Common determiners
    static const std::unordered_set<std::string> determiners = {
        "the", "a", "an",                     // Articles
        "this", "that", "these", "those",     // Demonstratives
        "my", "your", "his", "her", "its",    // Possessives
        "our", "their",
        "any", "many", "much", "few",         // Quantifiers
        "several", "some", "all", "both",
        "each", "every", "either", "neither",
        "no", "other", "another"
    };
    
    // Convert token to lowercase for comparison
    std::string lower_token = token;
    std::transform(lower_token.begin(), lower_token.end(), lower_token.begin(), ::tolower);
    
    return determiners.find(lower_token) != determiners.end();
}

bool Tokenizer::is_verb(const std::string& token) const {
    // Common verb endings
    static const std::vector<std::string> verb_endings = {
        "ing", "ed", "ate", "ize", "ify", "ise", "ect",
        "ent", "age", "ute", "end", "ish", "ade", "ine",
        "ume", "ure", "ide", "ive", "ete", "act"
    };

    // Common verbs that don't follow standard patterns
    static const std::unordered_set<std::string> common_verbs = {
        "go", "do", "make", "take", "come", "see", "get",
        "know", "find", "give", "tell", "work", "call", "try",
        "ask", "need", "feel", "let", "put", "mean", "keep",
        "run", "set", "move", "play", "pay", "hear", "help",
        "talk", "turn", "start", "show", "wait", "plan", "learn",
        "be", "am", "is", "are", "was", "were", "being", "been",
        "have", "has", "had", "having", "do", "does", "did", "doing",
        "say", "says", "said", "saying", "think", "thinks", "thought",
        "want", "wants", "wanted", "use", "uses", "used", "look",
        "looks", "looked", "find", "finds", "found", "give", "gives",
        "gave", "tell", "tells", "told", "work", "works", "worked",
        "call", "calls", "called", "try", "tries", "tried"
    };

    // Convert token to lowercase for comparison
    std::string lower_token = token;
    std::transform(lower_token.begin(), lower_token.end(), lower_token.begin(), ::tolower);
    
    // First check if it's a common verb
    if (common_verbs.find(lower_token) != common_verbs.end()) {
        return true;
    }

    // Then check for verb endings
    for (const auto& ending : verb_endings) {
        if (lower_token.length() > ending.length() && 
            lower_token.substr(lower_token.length() - ending.length()) == ending) {
            return true;
        }
    }

    return false;
}

void Tokenizer::initialize(const std::string& encoding_name) {
    auto tiktoken = std::make_unique<TiktokenTokenizer>();
    tiktoken->initialize(encoding_name);
    tokenizer_ = std::move(tiktoken);
}

// Add constructor implementation
Tokenizer::Tokenizer(const std::string& encoding_name) {
    initialize(encoding_name);
}

// Add vocab_size implementation
size_t Tokenizer::vocab_size() const {
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    return tokenizer_->vocab_size();
}