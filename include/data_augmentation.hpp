#pragma once

#include <string>
#include <vector>
#include <utility>
#include <random>
#include "tokenizer.hpp"

class DataAugmentation {
public:
    DataAugmentation(float p_synonym = 0.3f, float p_back_translation = 0.3f);

    /**
     * @brief Augment a single training pair
     * @param input Original input text
     * @param output Original output text
     * @return Vector of augmented pairs
     */
    std::vector<std::pair<std::string, std::string>> augmentPair(
        const std::string& input, const std::string& output);

    /**
     * @brief Augment entire dataset
     * @param training_pairs Original training pairs
     * @return Augmented dataset
     */
    std::vector<std::pair<std::string, std::string>> augmentDataset(
        const std::vector<std::pair<std::string, std::string>>& training_pairs);

    std::string augment_sequence(const std::string& sequence);

private:
    float p_synonym_;        // Probability of applying synonym replacement
    float p_back_translation_; // Probability of applying back translation
    std::mt19937 rng_;      // Random number generator

    // Augmentation techniques
    std::pair<std::string, std::string> synonymReplacement(
        const std::string& input, const std::string& output);
    std::pair<std::string, std::string> backTranslation(
        const std::string& input, const std::string& output);
    std::string insertNoise(const std::string& text);
}; 