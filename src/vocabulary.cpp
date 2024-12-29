#include "../include/vocabulary.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>

Vocabulary::Vocabulary() {
  // Initialize special tokens first (guaranteed IDs)
  add_special_token("<pad>", 0);  // Padding token
  add_special_token("<unk>", 1);  // Unknown token
  add_special_token("<bos>", 2);  // Beginning of sequence
  add_special_token("<eos>", 3);  // End of sequence
  add_special_token("<mask>", 4); // Mask token for MLM

  // Store special token IDs
  pad_token_id = 0;
  unk_token_id = 1;
  bos_token_id = 2;
  eos_token_id = 3;

  initialize_basic_vocabulary();
}

void Vocabulary::add_word(const std::string &word) {
  if (token_to_id.find(word) == token_to_id.end()) {
    int id = id_to_token.size();
    token_to_id[word] = id;
    id_to_token.push_back(word);
  }
}

void Vocabulary::add_special_token(const std::string &token, int id) {
  token_to_id[token] = id;
  if (id >= id_to_token.size()) {
    id_to_token.resize(id + 1);
  }
  id_to_token[id] = token;
}

void Vocabulary::initialize_basic_vocabulary() {
  // Basic pronouns and their contractions
  std::vector<std::string> pronouns = {
      "i",         "me",     "my",       "mine",  "myself",  "you",
      "your",      "yours",  "yourself", "he",    "him",     "his",
      "himself",   "she",    "her",      "hers",  "herself", "it",
      "its",       "itself", "we",       "us",    "our",     "ours",
      "ourselves", "they",   "them",     "their", "theirs",  "themselves"};

  // Common contractions
  std::vector<std::string> contractions = {
      "i'm",     "i've",    "i'll",    "i'd",   "you're", "you've", "you'll",
      "you'd",   "he's",    "he'll",   "he'd",  "she's",  "she'll", "she'd",
      "it's",    "it'll",   "it'd",    "we're", "we've",  "we'll",  "we'd",
      "they're", "they've", "they'll", "they'd"};

  // Common verbs
  std::vector<std::string> verbs = {
      "be",  "am",  "is",     "are", "was",  "were", "being", "been", "have",
      "has", "had", "having", "do",  "does", "did",  "doing", "done"};

  // Add all words to vocabulary
  std::vector<std::string> all_words;
  all_words.insert(all_words.end(), pronouns.begin(), pronouns.end());
  all_words.insert(all_words.end(), contractions.begin(), contractions.end());
  all_words.insert(all_words.end(), verbs.begin(), verbs.end());

  // Sort and remove duplicates
  std::sort(all_words.begin(), all_words.end());
  all_words.erase(std::unique(all_words.begin(), all_words.end()),
                  all_words.end());

  // Add each word
  for (const auto &word : all_words) {
    add_word(word);
  }
}

int Vocabulary::get_id(const std::string &token) const {
  auto it = token_to_id.find(token);
  return it != token_to_id.end() ? it->second : unk_token_id;
}

std::string Vocabulary::get_token(int id) const {
  return (id >= 0 && id < id_to_token.size()) ? id_to_token[id]
                                              : id_to_token[unk_token_id];
}

size_t Vocabulary::size() const { return id_to_token.size(); }

void Vocabulary::print_vocabulary_mappings() const {
  std::cout << "\n=== Special Tokens ===\n";
  std::cout << "PAD token: " << pad_token_id << " <-> "
            << id_to_token[pad_token_id] << "\n";
  std::cout << "UNK token: " << unk_token_id << " <-> "
            << id_to_token[unk_token_id] << "\n";
  std::cout << "BOS token: " << bos_token_id << " <-> "
            << id_to_token[bos_token_id] << "\n";
  std::cout << "EOS token: " << eos_token_id << " <-> "
            << id_to_token[eos_token_id] << "\n";

  std::cout << "\n=== Full Vocabulary ===\n";
  std::cout << "Total size: " << size() << " tokens\n\n";
  for (const auto &[token, id] : token_to_id) {
    std::cout << "ID " << std::setw(3) << id << ": '" << std::setw(15) << token
              << "'\n";
  }
}

bool Vocabulary::verify_mappings() const {
  bool valid = true;
  for (const auto &[token, id] : token_to_id) {
    if (id >= id_to_token.size()) {
      std::cout << "Error: Token '" << token << "' maps to invalid ID " << id
                << "\n";
      valid = false;
      continue;
    }
    if (id_to_token[id] != token) {
      std::cout << "Error: Inconsistent mapping for token '" << token << "'\n";
      std::cout << "token_to_id['" << token << "'] = " << id << "\n";
      std::cout << "id_to_token[" << id << "] = '" << id_to_token[id] << "'\n";
      valid = false;
    }
  }
  return valid;
}