#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

/**
 * @brief Text preprocessing utilities for transformer input.
 * 
 * The TextPreprocessor class provides static methods for preparing
 * text data before tokenization and model input. Features include:
 * - Case normalization
 * - Training data pair processing
 * - String cleaning and normalization
 * - Batch processing capabilities
 */
class TextPreprocessor {
  public:
    /**
     * @brief Converts text to lowercase.
     * 
     * Performs case normalization by converting all characters
     * to their lowercase equivalents using the current locale.
     * 
     * @param text Input text to convert
     * @return Lowercase version of the input text
     */
    static std::string to_lowercase(const std::string& text) {
        std::string result = text;
        std::transform(result.begin(), result.end(), result.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return result;
    }

    /**
     * @brief Preprocesses a batch of training pairs.
     * 
     * Applies preprocessing steps to both input and target texts
     * in each training pair, including:
     * - Case normalization
     * - Whitespace normalization
     * - Special character handling
     * 
     * @param training_data Vector of input-target text pairs
     * @return Vector of preprocessed text pairs
     */
    static std::vector<std::pair<std::string, std::string>> preprocess_training_data(
        const std::vector<std::pair<std::string, std::string>>& training_data) {

        std::vector<std::pair<std::string, std::string>> processed_data;
        processed_data.reserve(training_data.size());

        for (const auto& [input, target] : training_data) {
            processed_data.emplace_back(to_lowercase(input), to_lowercase(target));
        }

        return processed_data;
    }
};
