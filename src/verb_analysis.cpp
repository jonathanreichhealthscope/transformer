#include "../include/verb_analysis.hpp"
#include <sstream>

// Initialize static members
const std::vector<std::string> VerbPhraseAnalyzer::verb_suffixes = {
    "ate", "ize", "ify", "ise", "ect", "ent", "age", "ute"
};

const std::unordered_set<std::string> VerbPhraseAnalyzer::common_verbs = {
    // Organization/Management
    "organize", "manage", "coordinate", "direct", "lead", "guide", "plan", "arrange", "structure",
    "architect", "design", "develop", "implement", "execute", "oversee", "supervise", "mentor", "teach",
    // Communication
    "communicate", "speak", "write", "present", "explain", "discuss", "describe", "inform", "report",
    "announce", "state", "declare", "express", "convey", "relate", "brief", "instruct",
    // Analysis
    "analyze", "evaluate", "assess", "review", "examine", "study", "investigate", "research", "explore",
    "inspect", "audit", "monitor", "track", "measure", "calculate", "compute", "determine",
    // Creation
    "create", "build", "make", "construct", "produce", "generate", "develop", "establish", "form",
    "compose", "design", "craft", "fabricate", "manufacture", "assemble", "prepare",
    // Problem Solving
    "solve", "resolve", "fix", "repair", "troubleshoot", "debug", "address", "handle", "process",
    "improve", "enhance", "optimize", "streamline", "simplify", "clarify", "correct"
};

bool VerbPhraseAnalyzer::isVerb(const std::string& word) {
    // First check if it's in our common verbs list
    if (common_verbs.find(word) != common_verbs.end()) {
        return true;
    }
    
    // Check for verb suffixes
    for (const auto& suffix : verb_suffixes) {
        if (word.length() > suffix.length() && 
            word.substr(word.length() - suffix.length()) == suffix) {
            return true;
        }
    }
    return false;
}

bool VerbPhraseAnalyzer::endsWithVerb(const std::string& phrase) {
    std::istringstream iss(phrase);
    std::vector<std::string> words;
    std::string word;
    
    // Split phrase into words
    while (iss >> word) {
        words.push_back(word);
    }
    
    // Check if the last word is a verb
    if (!words.empty()) {
        return isVerb(words.back());
    }
    
    return false;
}

std::vector<std::string> VerbPhraseAnalyzer::extractVerbPhrases(const std::string& filename) {
    std::vector<std::string> phrases;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty()) continue;

        // Process marked verbs in training data
        if (hasMarkedVerb(line)) {
            processTrainingLine(line);
            phrases.push_back(line);  // Add the full pattern for training
        }
        // Also check for regular verb phrases
        else if (endsWithVerb(line)) {
            phrases.push_back(line);
        }
    }

    file.close();
    return phrases;
}

void VerbPhraseAnalyzer::analyzeAndLogPhrases(const std::vector<std::string>& phrases, std::ofstream& log_file) {
    // Count unique phrases
    std::set<std::string> unique_phrases(phrases.begin(), phrases.end());
    
    // Count frequencies
    std::map<std::string, int> phrase_frequencies;
    for (const auto& phrase : phrases) {
        phrase_frequencies[phrase]++;
    }
    
    // Sort by frequency
    std::vector<std::pair<std::string, int>> sorted_phrases(
        phrase_frequencies.begin(), phrase_frequencies.end()
    );
    std::sort(sorted_phrases.begin(), sorted_phrases.end(),
             [](const auto& a, const auto& b) {
                 return a.second > b.second;
             });
    
    // Log unique phrases count
    log_file << "Found " << unique_phrases.size() << " unique verb phrases\n\n";
    
    // Log top 10 most common phrases
    log_file << "Top 10 most common verb phrases:\n";
    for (size_t i = 0; i < std::min(size_t(10), sorted_phrases.size()); ++i) {
        log_file << sorted_phrases[i].first << ": " 
                << sorted_phrases[i].second << " occurrences\n";
    }
    log_file << "\n";
}

std::string VerbPhraseAnalyzer::trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

// Extract marked verb from training pattern (e.g., "She wishes to # architect" -> "architect")
std::string VerbPhraseAnalyzer::extractMarkedVerb(const std::string& line) {
    size_t hash_pos = line.find('#');
    if (hash_pos == std::string::npos) return "";
    
    // Get the word after the hash
    std::string rest = line.substr(hash_pos + 1);
    std::istringstream iss(rest);
    std::string word;
    iss >> word;  // Get first word after hash
    return word;
}

// Check if a line contains a marked verb pattern
bool VerbPhraseAnalyzer::hasMarkedVerb(const std::string& line) {
    return line.find('#') != std::string::npos;
}

// Process training line and add to verb set if marked
void VerbPhraseAnalyzer::processTrainingLine(const std::string& line) {
    if (hasMarkedVerb(line)) {
        std::string verb = extractMarkedVerb(line);
        if (!verb.empty()) {
            common_verbs.insert(verb);
        }
    }
} 