#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

class TextPreprocessor {
public:
  // Convert a single string to lowercase
  static std::string to_lowercase(const std::string &text) {
    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
  }

  // Process a vector of training pairs
  static std::vector<std::pair<std::string, std::string>>
  preprocess_training_data(
      const std::vector<std::pair<std::string, std::string>> &training_data) {

    std::vector<std::pair<std::string, std::string>> processed_data;
    processed_data.reserve(training_data.size());

    for (const auto &[input, target] : training_data) {
      processed_data.emplace_back(to_lowercase(input), to_lowercase(target));
    }

    return processed_data;
  }
};
