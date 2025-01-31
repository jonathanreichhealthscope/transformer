#pragma once

#include <string>
#include <set>
#include <vector>

namespace transformer {

class VocabularyCounter {
public:
    // Count unique tokens from both training and validation files
    static size_t countUniqueTokens(const std::string& training_file, const std::string& validation_file);

private:
    // Helper method to read and process a single file
    static void processFile(const std::string& filename, std::set<std::string>& unique_tokens);
};

} // namespace transformer 