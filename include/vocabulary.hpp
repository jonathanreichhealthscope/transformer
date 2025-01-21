#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class Vocabulary {
  private:
    std::unordered_map<std::string, int> token_to_id;
    std::vector<std::string> id_to_token;
    std::unordered_set<std::string> nouns;
    int unk_token_id;
    int pad_token_id;
    int bos_token_id;
    int eos_token_id;
    int mask_token_id;

    void add_word(const std::string& word);

  public:
    Vocabulary();
    void add_special_token(const std::string& token, int id);
    void initialize_basic_vocabulary();
    int get_id(const std::string& token) const;
    std::string get_token(int id) const;
    size_t size() const;

    // Utility methods
    int get_pad_token_id() const {
        return pad_token_id;
    }
    int get_unk_token_id() const {
        return unk_token_id;
    }
    int get_bos_token_id() const {
        return bos_token_id;
    }
    int get_eos_token_id() const {
        return eos_token_id;
    }
    int get_mask_token_id() const {
        return mask_token_id;
    }

    void print_vocabulary_mappings() const;
    bool verify_mappings() const;

    bool has_token(const std::string& token) const {
        return token_to_id.find(token) != token_to_id.end();
    }

    bool is_noun(const std::string& token) const;
    void load_nouns(const std::string& noun_file_path);
};