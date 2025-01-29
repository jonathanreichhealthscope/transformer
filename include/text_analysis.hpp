#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <fstream>
#include <unordered_set>

class AdjectivePhraseAnalyzer {
public:
    static std::vector<std::string> extractAdjectivePhrases(const std::string& filename);
    static void analyzeAndLogPhrases(const std::vector<std::string>& phrases, std::ofstream& log_file);

private:
    static std::string trim(const std::string& str);
    static bool isAdjective(const std::string& word);
    static bool endsWithAdjective(const std::string& phrase);
    
    // Common adjective suffixes
    static const std::vector<std::string> adjective_suffixes;
    
    // Common adjectives (partial list - can be expanded)
    static const std::unordered_set<std::string> common_adjectives;
}; 