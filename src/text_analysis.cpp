#include "../include/text_analysis.hpp"
#include <sstream>

// Initialize static members
const std::vector<std::string> AdjectivePhraseAnalyzer::adjective_suffixes = {
    "able", "ible", "al", "ful", "ic", "ive", "less", "ous", "y", "ent", "ant"
};

const std::unordered_set<std::string> AdjectivePhraseAnalyzer::common_adjectives = {
    // Colors
    "red", "blue", "green", "yellow", "black", "white", "purple", "orange", "brown", "grey", "gray",
    // Sizes
    "big", "small", "large", "tiny", "huge", "massive", "little", "giant", "enormous", "miniature",
    // Qualities
    "good", "bad", "great", "poor", "excellent", "terrible", "wonderful", "awful", "perfect", "horrible",
    // States
    "hot", "cold", "warm", "cool", "wet", "dry", "clean", "dirty", "fresh", "stale",
    // Emotions
    "happy", "sad", "angry", "calm", "excited", "bored", "tired", "energetic", "peaceful", "anxious",
    // Appearances
    "beautiful", "ugly", "pretty", "handsome", "gorgeous", "plain", "elegant", "shabby", "attractive", "unattractive",
    // Textures
    "smooth", "rough", "soft", "hard", "silky", "coarse", "bumpy", "fluffy", "crisp", "fuzzy",
    // Shapes
    "round", "square", "oval", "rectangular", "circular", "triangular", "flat", "curved", "straight", "crooked",
    // Ages
    "new", "old", "young", "ancient", "modern", "fresh", "aged", "recent", "vintage", "contemporary",
    // Values
    "expensive", "cheap", "valuable", "worthless", "precious", "priceless", "costly", "affordable", "reasonable", "overpriced",
    // Intelligence
    "smart", "clever", "intelligent", "wise", "brilliant", "dumb", "stupid", "slow", "sharp", "bright",
    // Speed
    "fast", "slow", "quick", "rapid", "swift", "sluggish", "speedy", "leisurely", "hasty", "prompt",
    // Common states
    "open", "closed", "empty", "full", "light", "heavy", "deep", "shallow", "thick", "thin",
    // Difficulty
    "easy", "hard", "difficult", "simple", "complex", "complicated", "straightforward", "challenging", "demanding", "effortless",
    // Importance
    "important", "unimportant", "crucial", "vital", "essential", "trivial", "significant", "minor", "major", "critical",
    // Frequency
    "common", "rare", "frequent", "occasional", "regular", "unusual", "unique", "typical", "normal", "strange",
    // Certainty
    "sure", "uncertain", "definite", "clear", "obvious", "vague", "apparent", "evident", "doubtful", "questionable",
    // Necessity
    "necessary", "unnecessary", "needed", "optional", "required", "essential", "vital", "dispensable", "mandatory", "voluntary",
    // Completeness
    "complete", "incomplete", "whole", "partial", "finished", "unfinished", "total", "entire", "full", "partial"
};

bool AdjectivePhraseAnalyzer::isAdjective(const std::string& word) {
    // First check if it's in our common adjectives list
    if (common_adjectives.find(word) != common_adjectives.end()) {
        return true;
    }
    
    // Check for adjective suffixes
    for (const auto& suffix : adjective_suffixes) {
        if (word.length() > suffix.length() && 
            word.substr(word.length() - suffix.length()) == suffix) {
            return true;
        }
    }
    return false;
}

bool AdjectivePhraseAnalyzer::endsWithAdjective(const std::string& phrase) {
    // If it's a marked adjective pattern, extract and check the marked word
    if (hasMarkedAdjective(phrase)) {
        std::string adj = extractMarkedAdjective(phrase);
        return !adj.empty() && isAdjective(adj);
    }

    // Otherwise check the last word
    std::istringstream iss(phrase);
    std::vector<std::string> words;
    std::string word;
    
    while (iss >> word) {
        words.push_back(word);
    }
    
    if (!words.empty()) {
        return isAdjective(words.back());
    }
    
    return false;
}

std::vector<std::string> AdjectivePhraseAnalyzer::extractAdjectivePhrases(const std::string& filename) {
    std::vector<std::string> phrases;
    std::vector<std::string> marked_adjective_phrases;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty()) continue;

        // Process marked adjectives in training data
        if (hasMarkedAdjective(line)) {
            std::string adj = extractMarkedAdjective(line);
            if (!adj.empty() && isAdjective(adj)) {
                processTrainingLine(line);
                marked_adjective_phrases.push_back(line);
            }
        }
        // Also check for regular adjective phrases
        else if (endsWithAdjective(line)) {
            phrases.push_back(line);
        }
    }

    file.close();
    
    // Combine both types of phrases, with marked adjectives first
    std::vector<std::string> all_phrases;
    all_phrases.insert(all_phrases.end(), marked_adjective_phrases.begin(), marked_adjective_phrases.end());
    all_phrases.insert(all_phrases.end(), phrases.begin(), phrases.end());
    return all_phrases;
}

void AdjectivePhraseAnalyzer::analyzeAndLogPhrases(const std::vector<std::string>& phrases, std::ofstream& log_file) {
    // Separate marked and unmarked adjective phrases
    std::vector<std::string> marked_phrases;
    std::vector<std::string> regular_phrases;
    
    for (const auto& phrase : phrases) {
        if (hasMarkedAdjective(phrase)) {
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
    log_file << "Adjective Phrase Statistics:\n";
    log_file << "- Marked adjective phrases: " << unique_marked.size() << "\n";
    log_file << "- Regular adjective phrases: " << unique_regular.size() << "\n";
    log_file << "- Total unique adjective phrases: " << unique_marked.size() + unique_regular.size() << "\n\n";
    
    // Log top 10 most common phrases
    log_file << "Top 10 most common adjective phrases:\n";
    for (size_t i = 0; i < std::min(size_t(10), sorted_phrases.size()); ++i) {
        log_file << sorted_phrases[i].first << ": " 
                << sorted_phrases[i].second << " occurrences\n";
    }
    log_file << "\n";
    
    // Log some examples of marked adjectives
    if (!marked_phrases.empty()) {
        log_file << "Examples of marked adjective phrases:\n";
        size_t example_count = std::min(size_t(5), marked_phrases.size());
        for (size_t i = 0; i < example_count; ++i) {
            log_file << "- " << marked_phrases[i] << "\n";
        }
        log_file << "\n";
    }
}

std::string AdjectivePhraseAnalyzer::trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

// Extract marked adjective from training pattern (e.g., "The sky is * blue" -> "blue")
std::string AdjectivePhraseAnalyzer::extractMarkedAdjective(const std::string& line) {
    size_t asterisk_pos = line.find('*');
    if (asterisk_pos == std::string::npos) return "";
    
    // Get the word after the asterisk
    std::string rest = line.substr(asterisk_pos + 1);
    std::istringstream iss(rest);
    std::string word;
    iss >> word;  // Get first word after asterisk
    
    // Remove any trailing punctuation
    word.erase(std::remove_if(word.begin(), word.end(), 
        [](char c) { return std::ispunct(c); }), word.end());
    
    return word;
}

// Check if a line contains a marked adjective pattern
bool AdjectivePhraseAnalyzer::hasMarkedAdjective(const std::string& line) {
    return line.find('*') != std::string::npos;
}

// Process training line and add to adjective set if marked
void AdjectivePhraseAnalyzer::processTrainingLine(const std::string& line) {
    if (hasMarkedAdjective(line)) {
        std::string adj = extractMarkedAdjective(line);
        if (!adj.empty()) {
            // Add both the original form and a lowercase version
            common_adjectives.insert(adj);
            std::string lower_adj = adj;
            std::transform(lower_adj.begin(), lower_adj.end(), lower_adj.begin(), ::tolower);
            common_adjectives.insert(lower_adj);
        }
    }
} 