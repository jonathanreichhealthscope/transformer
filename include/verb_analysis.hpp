#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <fstream>
#include <unordered_set>

class VerbPhraseAnalyzer {
public:
    static std::vector<std::string> extractVerbPhrases(const std::string& filename);
    static void analyzeAndLogPhrases(const std::vector<std::string>& phrases, std::ofstream& log_file);
    static std::string extractMarkedVerb(const std::string& line);
    static bool hasMarkedVerb(const std::string& line);
    static void processTrainingLine(const std::string& line);
    static bool isVerb(const std::string& word);

private:
    static std::string trim(const std::string& str);
    static bool endsWithVerb(const std::string& phrase);
    
    // Common verb suffixes
    static const std::vector<std::string> verb_suffixes;
    
    // Common verbs (partial list - can be expanded)
    static std::unordered_set<std::string> common_verbs;
}; 