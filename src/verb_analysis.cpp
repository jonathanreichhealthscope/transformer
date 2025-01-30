#include "../include/verb_analysis.hpp"
#include <sstream>
#include <cctype>

// Initialize static members
const std::vector<std::string> VerbPhraseAnalyzer::verb_suffixes = {
    "ate", "ize", "ify", "ise", "ect", "ent", "age", "ute", "ing", "ed", "es", "s"
};

std::unordered_set<std::string> VerbPhraseAnalyzer::common_verbs = {
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
    // If it's a marked verb pattern, extract and check the marked word
    if (hasMarkedVerb(phrase)) {
        std::string verb = extractMarkedVerb(phrase);
        return !verb.empty() && isVerb(verb);
    }

    // Otherwise check the last word
    std::istringstream iss(phrase);
    std::vector<std::string> words;
    std::string word;
    
    while (iss >> word) {
        words.push_back(word);
    }
    
    if (!words.empty()) {
        return isVerb(words.back());
    }
    
    return false;
}

std::vector<std::string> VerbPhraseAnalyzer::extractVerbPhrases(const std::string& filename) {
    std::vector<std::string> phrases;
    std::vector<std::string> marked_verb_phrases;
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
            std::string verb = extractMarkedVerb(line);
            if (!verb.empty() && isVerb(verb)) {
                processTrainingLine(line);
                marked_verb_phrases.push_back(line);
            }
        }
        // Also check for regular verb phrases
        else if (endsWithVerb(line)) {
            phrases.push_back(line);
        }
    }

    file.close();
    
    // Combine both types of phrases, with marked verbs first
    std::vector<std::string> all_phrases;
    all_phrases.insert(all_phrases.end(), marked_verb_phrases.begin(), marked_verb_phrases.end());
    all_phrases.insert(all_phrases.end(), phrases.begin(), phrases.end());
    return all_phrases;
}

void VerbPhraseAnalyzer::analyzeAndLogPhrases(const std::vector<std::string>& phrases, std::ofstream& log_file) {
    // Separate marked and unmarked verb phrases
    std::vector<std::string> marked_phrases;
    std::vector<std::string> regular_phrases;
    
    for (const auto& phrase : phrases) {
        if (hasMarkedVerb(phrase)) {
            marked_phrases.push_back(phrase);
        } else {
            regular_phrases.push_back(phrase);
        }
    }
    
    // Count unique phrases
    std::set<std::string> unique_marked(marked_phrases.begin(), marked_phrases.end());
    std::set<std::string> unique_regular(regular_phrases.begin(), regular_phrases.end());
    
    // Count frequencies for all phrases
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
    
    // Log statistics
    log_file << "Verb Phrase Statistics:\n";
    log_file << "- Marked verb phrases: " << unique_marked.size() << "\n";
    log_file << "- Regular verb phrases: " << unique_regular.size() << "\n";
    log_file << "- Total unique verb phrases: " << unique_marked.size() + unique_regular.size() << "\n\n";
    
    // Log top 10 most common phrases
    log_file << "Top 10 most common verb phrases:\n";
    for (size_t i = 0; i < std::min(size_t(10), sorted_phrases.size()); ++i) {
        log_file << sorted_phrases[i].first << ": " 
                << sorted_phrases[i].second << " occurrences\n";
    }
    log_file << "\n";
    
    // Log some examples of marked verbs
    if (!marked_phrases.empty()) {
        log_file << "Examples of marked verb phrases:\n";
        size_t example_count = std::min(size_t(5), marked_phrases.size());
        for (size_t i = 0; i < example_count; ++i) {
            log_file << "- " << marked_phrases[i] << "\n";
        }
        log_file << "\n";
    }
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
    
    // Remove any trailing punctuation
    word.erase(std::remove_if(word.begin(), word.end(), 
        [](char c) { return std::ispunct(c); }), word.end());
    
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
            // Add both the original form and a lowercase version
            common_verbs.insert(verb);
            std::string lower_verb = verb;
            std::transform(lower_verb.begin(), lower_verb.end(), lower_verb.begin(), ::tolower);
            common_verbs.insert(lower_verb);
        }
    }
} 