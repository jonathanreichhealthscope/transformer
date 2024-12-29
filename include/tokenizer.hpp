#pragma once
#include "vocabulary.hpp"
#include <memory>
#include <sstream>

class Tokenizer {
private:
  std::unique_ptr<Vocabulary> vocab;

  void save_vocabulary(std::ostream &os) const;
  static std::unique_ptr<Vocabulary> load_vocabulary(std::istream &is);

public:
  Tokenizer();
  std::vector<int> encode(const std::string &text) const;
  std::string decode(const std::vector<int> &tokens) const;
  void save(std::ostream &os) const;
  static std::unique_ptr<Tokenizer> load(std::istream &is);
  size_t vocab_size() const { return vocab->size(); }
  bool is_special_token(int token_id) const;

  void print_vocabulary_mappings() const { vocab->print_vocabulary_mappings(); }
  bool verify_mappings() const { return vocab->verify_mappings(); }
};