#include "../include/tokenizer.hpp"
#include <fstream>

Tokenizer::Tokenizer(size_t vocab_size_) : vocab_size(vocab_size_) {
  // Initialize with basic vocabulary
  vocab.resize(vocab_size);
  for (size_t i = 0; i < vocab_size; ++i) {
    vocab[i] = "<token" + std::to_string(i) + ">";
  }
}

std::vector<int> Tokenizer::encode(const std::string &text) {
  // Simple character-based tokenization for now
  std::vector<int> tokens;
  for (char c : text) {
    // Map character to token ID (simple mapping for demonstration)
    int token_id = static_cast<int>(c) % vocab_size;
    tokens.push_back(token_id);
  }
  return tokens;
}

std::string Tokenizer::decode(const std::vector<int> &tokens) {
  std::string text;
  for (int token_id : tokens) {
    if (token_id >= 0 && static_cast<size_t>(token_id) < vocab_size) {
      text += vocab[token_id];
    }
  }
  return text;
}

void Tokenizer::save(std::ostream &os) const {
  // Save vocab size
  os.write(reinterpret_cast<const char *>(&vocab_size), sizeof(vocab_size));

  // Save vocabulary
  for (const auto &token : vocab) {
    size_t len = token.length();
    os.write(reinterpret_cast<const char *>(&len), sizeof(len));
    os.write(token.c_str(), len);
  }
}

std::unique_ptr<Tokenizer> Tokenizer::load(std::istream &is) {
  size_t vocab_size;
  is.read(reinterpret_cast<char *>(&vocab_size), sizeof(vocab_size));

  auto tokenizer = std::make_unique<Tokenizer>(vocab_size);

  // Load vocabulary
  for (size_t i = 0; i < vocab_size; ++i) {
    size_t len;
    is.read(reinterpret_cast<char *>(&len), sizeof(len));
    std::string token(len, '\0');
    is.read(&token[0], len);
    tokenizer->vocab[i] = token;
  }

  return tokenizer;
}