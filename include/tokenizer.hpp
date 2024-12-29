#pragma once
#include <memory>
#include <string>
#include <vector>

class Tokenizer {
private:
  std::vector<std::string> vocab;
  size_t vocab_size;

public:
  Tokenizer(size_t vocab_size = 50000);
  std::vector<int> encode(const std::string &text);
  std::string decode(const std::vector<int> &tokens);
  void save(std::ostream &os) const;
  static std::unique_ptr<Tokenizer> load(std::istream &is);
};