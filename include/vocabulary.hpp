#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * @brief Manages the vocabulary for the tokenizer, including special tokens and mappings.
 * 
 * The Vocabulary class maintains bidirectional mappings between tokens (strings) and
 * their corresponding IDs (integers). It handles:
 * - Special tokens (PAD, UNK, BOS, EOS, MASK)
 * - Basic vocabulary initialization
 * - Token-to-ID and ID-to-token conversions
 * - Noun identification for linguistic tasks
 */
class Vocabulary {
  private:
    std::unordered_map<std::string, int> token_to_id;  ///< Maps tokens to their unique IDs
    std::vector<std::string> id_to_token;              ///< Maps IDs back to their tokens
    std::unordered_set<std::string> nouns;             ///< Set of tokens identified as nouns
    int unk_token_id;                                  ///< ID for unknown token
    int pad_token_id;                                  ///< ID for padding token
    int bos_token_id;                                  ///< ID for beginning-of-sequence token
    int eos_token_id;                                  ///< ID for end-of-sequence token
    int mask_token_id;                                 ///< ID for mask token

    /**
     * @brief Adds a word to the vocabulary.
     * @param word Word to add
     */
    void add_word(const std::string& word);

  public:
    /**
     * @brief Default constructor.
     * Initializes special token IDs and empty mappings.
     */
    Vocabulary();

    /**
     * @brief Adds a special token with a specific ID.
     * @param token Special token to add
     * @param id ID to assign to the token
     */
    void add_special_token(const std::string& token, int id);

    /**
     * @brief Initializes the vocabulary with basic tokens.
     * This includes common words and special tokens.
     */
    void initialize_basic_vocabulary();

    /**
     * @brief Gets the ID for a given token.
     * @param token Token to look up
     * @return ID of the token, or UNK token ID if not found
     */
    int get_id(const std::string& token) const;

    /**
     * @brief Gets the token for a given ID.
     * @param id ID to look up
     * @return Token string corresponding to the ID
     */
    std::string get_token(int id) const;

    /**
     * @brief Gets the size of the vocabulary.
     * @return Number of tokens in the vocabulary
     */
    size_t size() const;

    /**
     * @brief Gets the ID of the padding token.
     * @return Padding token ID
     */
    int get_pad_token_id() const {
        return pad_token_id;
    }

    /**
     * @brief Gets the ID of the unknown token.
     * @return Unknown token ID
     */
    int get_unk_token_id() const {
        return unk_token_id;
    }

    /**
     * @brief Gets the ID of the beginning-of-sequence token.
     * @return BOS token ID
     */
    int get_bos_token_id() const {
        return bos_token_id;
    }

    /**
     * @brief Gets the ID of the end-of-sequence token.
     * @return EOS token ID
     */
    int get_eos_token_id() const {
        return eos_token_id;
    }

    /**
     * @brief Gets the ID of the mask token.
     * @return Mask token ID
     */
    int get_mask_token_id() const {
        return mask_token_id;
    }

    /**
     * @brief Prints the current vocabulary mappings for debugging.
     */
    void print_vocabulary_mappings() const;

    /**
     * @brief Verifies the consistency of token-to-ID and ID-to-token mappings.
     * @return True if mappings are consistent
     */
    bool verify_mappings() const;

    /**
     * @brief Checks if a token exists in the vocabulary.
     * @param token Token to check
     * @return True if the token exists
     */
    bool has_token(const std::string& token) const {
        return token_to_id.find(token) != token_to_id.end();
    }

    /**
     * @brief Checks if a token is a noun.
     * @param token Token to check
     * @return True if the token is in the noun set
     */
    bool is_noun(const std::string& token) const;

    /**
     * @brief Loads a list of nouns from a file.
     * @param noun_file_path Path to the file containing nouns
     */
    void load_nouns(const std::string& noun_file_path);
};