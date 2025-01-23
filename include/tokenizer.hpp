#pragma once
#include "vocabulary.hpp"
#include "token_constants.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "sentencepiece_tokenizer.hpp"

/**
 * @brief Text tokenizer for converting between text and token sequences.
 * 
 * The Tokenizer class provides functionality for encoding text into token sequences
 * and decoding token sequences back into text. It handles:
 * - Special tokens (PAD, UNK, BOS, EOS, MASK)
 * - Text preprocessing
 * - Token caching for efficiency
 */
class Tokenizer {
  public:
    /**
     * @brief Default constructor.
     */
    Tokenizer();

    /**
     * @brief Encodes text into a sequence of token IDs.
     * @param text Input text to tokenize
     * @return Vector of token IDs
     */
    std::vector<int> encode(const std::string& text) const;

    /**
     * @brief Decodes a sequence of token IDs back into text.
     * @param tokens Vector of token IDs to decode
     * @return Decoded text string
     */
    std::string decode(const std::vector<int>& tokens) const;

    /**
     * @brief Preprocesses text before tokenization.
     * @param text Text to preprocess (modified in-place)
     */
    void preprocess_text(std::string& text) const;

    /**
     * @brief Checks if a token ID represents a special token.
     * @param token_id Token ID to check
     * @return True if the token is special (PAD, UNK, BOS, EOS, MASK)
     */
    bool is_special_token(int token_id) const;

    /**
     * @brief Saves the tokenizer state to a stream.
     * @param os Output stream to save to
     */
    void save(std::ostream& os) const;

    /**
     * @brief Loads a tokenizer from a stream.
     * @param is Input stream to load from
     * @return Unique pointer to the loaded tokenizer
     */
    static std::unique_ptr<Tokenizer> load(std::istream& is);

    /**
     * @brief Gets the size of the vocabulary.
     * @return Number of tokens in the vocabulary
     */
    size_t vocab_size() const {
        return vocab->size();
    }

    /**
     * @brief Prints the vocabulary token-to-ID mappings.
     */
    void print_vocabulary_mappings() const {
        vocab->print_vocabulary_mappings();
    }

    /**
     * @brief Verifies the consistency of token-to-ID mappings.
     * @return True if mappings are consistent
     */
    bool verify_mappings() const {
        return vocab->verify_mappings();
    }

    /**
     * @brief Checks if a token exists in the vocabulary.
     * @param token Token string to check
     * @return True if the token exists
     */
    bool has_token(const std::string& token) const {
        return vocab->has_token(token);
    }

    /**
     * @brief Clears the token encoding cache.
     */
    void clear_cache() {
        encoding_cache.clear();
    }

    /**
     * @brief Gets the ID of the padding token.
     * @return Padding token ID
     */
    int get_pad_token_id() const { return tokenizer_->get_pad_token_id(); }

    /**
     * @brief Gets the ID of the unknown token.
     * @return Unknown token ID
     */
    int get_unk_token_id() const { return tokenizer_->get_unk_token_id(); }

    /**
     * @brief Gets the ID of the beginning-of-sequence token.
     * @return BOS token ID
     */
    int get_bos_token_id() const { return tokenizer_->get_bos_token_id(); }

    /**
     * @brief Gets the ID of the end-of-sequence token.
     * @return EOS token ID
     */
    int get_eos_token_id() const { return tokenizer_->get_eos_token_id(); }

    /**
     * @brief Gets the ID of the mask token.
     * @return Mask token ID
     */
    int get_mask_token_id() const { return tokens::MASK_ID; }

    /**
     * @brief Gets a reference to the vocabulary.
     * @return Const reference to the vocabulary
     */
    const Vocabulary& get_vocabulary() const {
        return *vocab;
    }

    /**
     * @brief Gets the vocabulary as a vector.
     * @return Vector of tokens ordered by ID
     */
    std::vector<std::string> get_vocabulary_vector() const {
        return vocab->get_vocabulary_vector();
    }

    /**
     * @brief Gets the map of special characters to their string representations.
     * @return Map of special characters
     */
    static const std::unordered_map<char, std::string>& get_special_char_map() {
        return SPECIAL_CHAR_MAP;
    }

    void train(const std::vector<std::string>& texts, const std::string& model_prefix) {
        tokenizer_->train(texts, model_prefix);
    }

    void load_model(const std::string& model_path) {
        tokenizer_->load_model(model_path);
    }

  private:
    std::unique_ptr<Vocabulary> vocab;
    std::unique_ptr<SentencePieceTokenizer> tokenizer_;
    mutable std::unordered_map<std::string, std::vector<int>> encoding_cache;

    /// Map of special characters to their string representations
    static const std::unordered_map<char, std::string> SPECIAL_CHAR_MAP;

    /**
     * @brief Saves the vocabulary to a stream.
     * @param os Output stream to save to
     */
    void save_vocabulary(std::ostream& os) const;

    /**
     * @brief Loads the vocabulary from a stream.
     * @param is Input stream to load from
     * @return Unique pointer to the loaded vocabulary
     */
    static std::unique_ptr<Vocabulary> load_vocabulary(std::istream& is);

    /**
     * @brief Syncs the main vocabulary with the tokenizer's vocabulary.
     */
    void sync_vocabulary_with_subword_tokenizer();
};