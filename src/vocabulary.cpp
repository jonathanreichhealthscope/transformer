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
      "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
      "he", "him", "his", "himself", "she", "her", "hers", "herself",
      "it", "its", "itself", "we", "us", "our", "ours", "ourselves",
      "they", "them", "their", "theirs", "themselves", "this", "that",
      "these", "those", "who", "whom", "whose", "which", "what", "whatever",
      "whoever", "whomever", "anyone", "everyone", "someone", "nobody",
      "everybody", "somebody", "anyone", "everyone", "no one", "each", "either",
      "neither", "many", "few", "several", "all", "both", "any", "some"
  };

  // Common contractions and their variations
  std::vector<std::string> contractions = {
      "i'm", "i've", "i'll", "i'd", "you're", "you've", "you'll", "you'd",
      "he's", "he'll", "he'd", "she's", "she'll", "she'd", "it's", "it'll",
      "it'd", "we're", "we've", "we'll", "we'd", "they're", "they've",
      "they'll", "they'd", "isn't", "aren't", "wasn't", "weren't", "haven't",
      "hasn't", "hadn't", "doesn't", "don't", "didn't", "won't", "wouldn't",
      "can't", "couldn't", "mustn't", "shouldn't", "mightn't", "shan't",
      "let's", "that's", "who's", "what's", "here's", "there's", "where's",
      "when's", "why's", "how's", "daren't", "needn't", "oughtn't", "ain't"
  };

  // Common verbs with all their forms
  std::vector<std::string> verbs = {
      // Basic verbs
      "be", "am", "is", "are", "was", "were", "being", "been",
      "have", "has", "had", "having", "do", "does", "did", "doing", "done",
      // Common action verbs
      "go", "goes", "went", "going", "gone",
      "say", "says", "said", "saying",
      "get", "gets", "got", "getting", "gotten",
      "make", "makes", "made", "making",
      "know", "knows", "knew", "knowing", "known",
      "think", "thinks", "thought", "thinking",
      "take", "takes", "took", "taking", "taken",
      "see", "sees", "saw", "seeing", "seen",
      "come", "comes", "came", "coming",
      "want", "wants", "wanted", "wanting",
      "look", "looks", "looked", "looking",
      "use", "uses", "used", "using",
      "find", "finds", "found", "finding",
      "give", "gives", "gave", "giving", "given",
      "tell", "tells", "told", "telling",
      "work", "works", "worked", "working",
      "call", "calls", "called", "calling",
      "try", "tries", "tried", "trying",
      "ask", "asks", "asked", "asking",
      "need", "needs", "needed", "needing",
      "feel", "feels", "felt", "feeling",
      "become", "becomes", "became", "becoming",
      "leave", "leaves", "left", "leaving",
      "put", "puts", "putting",
      "mean", "means", "meant", "meaning",
      "keep", "keeps", "kept", "keeping",
      "let", "lets", "letting",
      "begin", "begins", "began", "beginning", "begun",
      "seem", "seems", "seemed", "seeming",
      "help", "helps", "helped", "helping",
      "talk", "talks", "talked", "talking",
      "turn", "turns", "turned", "turning",
      "show", "shows", "showed", "showing", "shown"
  };

  // Common prepositions and conjunctions
  std::vector<std::string> connectors = {
      // Prepositions
      "in", "on", "at", "to", "for", "with", "by", "from", "up", "about",
      "into", "over", "after", "beneath", "under", "above", "below", "behind",
      "between", "beyond", "during", "except", "through", "toward", "within",
      "without", "across", "along", "around", "before", "beside", "besides",
      "down", "inside", "near", "off", "since", "upon", "within", "throughout",
      // Conjunctions
      "and", "but", "or", "nor", "for", "yet", "so", "because", "although",
      "unless", "since", "while", "where", "if", "then", "else", "therefore",
      "however", "moreover", "furthermore", "nevertheless", "meanwhile",
      "afterwards", "consequently", "otherwise", "instead", "whereas"
  };

  // Common adjectives
  std::vector<std::string> adjectives = {
      "good", "new", "first", "last", "long", "great", "little", "own",
      "other", "old", "right", "big", "high", "different", "small", "large",
      "next", "early", "young", "important", "few", "public", "bad", "same",
      "able", "best", "better", "low", "late", "general", "specific", "certain",
      "free", "full", "special", "easy", "clear", "recent", "final", "main",
      "sure", "real", "available", "local", "particular", "hard", "major",
      "current", "nice", "happy", "serious", "ready", "simple", "possible",
      "whole", "short", "private", "past", "beautiful", "strong", "quick"
  };

  // Common nouns
  std::vector<std::string> nouns = {
      // People and roles
      "person", "people", "family", "friend", "parent", "mother", "father",
      "child", "baby", "teacher", "student", "doctor", "worker", "artist",
      
      // Places
      "home", "house", "school", "office", "store", "hospital", "city",
      "country", "world", "room", "building", "street", "park", "garden",
      
      // Time
      "time", "day", "night", "morning", "evening", "week", "month", "year",
      "today", "tomorrow", "minute", "hour", "moment", "future", "past",
      
      // Nature
      "water", "air", "earth", "fire", "sun", "moon", "star", "sky",
      "tree", "flower", "grass", "river", "ocean", "mountain", "forest",
      
      // Objects
      "book", "phone", "computer", "car", "door", "window", "table", "chair",
      "bed", "food", "money", "paper", "key", "screen", "picture",
      
      // Abstract concepts
      "life", "death", "love", "hate", "peace", "war", "truth", "lie",
      "idea", "thought", "dream", "hope", "fear", "mind", "soul",
      
      // Body parts
      "head", "face", "eye", "ear", "nose", "mouth", "hand", "foot",
      "heart", "brain", "body", "hair", "finger", "skin", "bone",
      
      // Groups and organizations
      "group", "team", "company", "government", "community", "society",
      "organization", "club", "party", "family"
  };

  // Add all words to vocabulary
  std::vector<std::string> all_words;
  all_words.insert(all_words.end(), pronouns.begin(), pronouns.end());
  all_words.insert(all_words.end(), contractions.begin(), contractions.end());
  all_words.insert(all_words.end(), verbs.begin(), verbs.end());
  all_words.insert(all_words.end(), connectors.begin(), connectors.end());
  all_words.insert(all_words.end(), adjectives.begin(), adjectives.end());
  all_words.insert(all_words.end(), nouns.begin(), nouns.end());

  // Sort and remove duplicates
  std::sort(all_words.begin(), all_words.end());
  all_words.erase(std::unique(all_words.begin(), all_words.end()), all_words.end());

  // Add each word
  for (const auto& word : all_words) {
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