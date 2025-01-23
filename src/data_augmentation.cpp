#include "../include/data_augmentation.hpp"
#include <algorithm>
#include <random>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>

// Comprehensive synonym dictionary organized by categories
static const std::unordered_map<std::string, std::vector<std::string>> SYNONYMS = {
    // Common adjectives
    {"good", {"great", "excellent", "fine", "nice", "wonderful", "fantastic", "superb", "outstanding"}},
    {"bad", {"poor", "terrible", "awful", "unpleasant", "horrible", "dreadful", "subpar", "inferior"}},
    {"happy", {"glad", "joyful", "pleased", "delighted", "cheerful", "content", "elated", "jubilant"}},
    {"sad", {"unhappy", "depressed", "down", "gloomy", "melancholy", "sorrowful", "dejected", "miserable"}},
    {"big", {"large", "huge", "enormous", "massive", "gigantic", "substantial", "extensive", "vast"}},
    {"small", {"tiny", "little", "miniature", "compact", "modest", "minute", "petite", "diminutive"}},

    // Emotions and feelings
    {"angry", {"furious", "enraged", "irate", "annoyed", "irritated", "outraged", "mad", "livid"}},
    {"scared", {"afraid", "frightened", "terrified", "fearful", "anxious", "startled", "petrified", "nervous"}},
    {"tired", {"exhausted", "weary", "fatigued", "drained", "sleepy", "worn-out", "spent", "drowsy"}},
    {"excited", {"thrilled", "enthusiastic", "eager", "animated", "energetic", "passionate", "zealous", "ardent"}},

    // Actions and verbs
    {"run", {"sprint", "dash", "jog", "race", "bolt", "rush", "hurry", "scamper"}},
    {"walk", {"stroll", "amble", "stride", "wander", "saunter", "trek", "march", "hike"}},
    {"say", {"tell", "speak", "utter", "express", "voice", "articulate", "communicate", "convey"}},
    {"look", {"gaze", "stare", "glance", "peek", "observe", "watch", "view", "examine"}},
    {"think", {"believe", "consider", "contemplate", "ponder", "reflect", "reason", "meditate", "deliberate"}},

    // Technical terms
    {"bug", {"error", "defect", "flaw", "glitch", "issue", "problem", "fault", "malfunction"}},
    {"fast", {"quick", "rapid", "swift", "speedy", "prompt", "expeditious", "brisk", "nimble"}},
    {"slow", {"sluggish", "leisurely", "unhurried", "gradual", "plodding", "dawdling", "tardy", "languid"}},
    {"new", {"recent", "fresh", "novel", "modern", "current", "contemporary", "latest", "innovative"}},
    {"old", {"ancient", "aged", "vintage", "antique", "traditional", "classic", "outdated", "obsolete"}},

    // Programming specific
    {"implement", {"develop", "create", "build", "construct", "code", "program", "design", "engineer"}},
    {"debug", {"troubleshoot", "fix", "resolve", "diagnose", "repair", "correct", "address", "solve"}},
    {"optimize", {"improve", "enhance", "refine", "streamline", "upgrade", "perfect", "tune", "polish"}},
    {"test", {"verify", "validate", "check", "examine", "assess", "evaluate", "analyze", "inspect"}},
    {"deploy", {"launch", "release", "publish", "distribute", "roll-out", "ship", "deliver", "implement"}},

    // Data science terms
    {"analyze", {"examine", "study", "investigate", "evaluate", "assess", "scrutinize", "review", "explore"}},
    {"predict", {"forecast", "project", "estimate", "anticipate", "foresee", "calculate", "determine", "extrapolate"}},
    {"process", {"handle", "manage", "execute", "perform", "conduct", "run", "operate", "accomplish"}},
    {"clean", {"sanitize", "prepare", "organize", "arrange", "structure", "format", "standardize", "normalize"}},
    {"validate", {"verify", "confirm", "check", "authenticate", "substantiate", "prove", "establish", "corroborate"}},

    // Common nouns
    {"problem", {"issue", "challenge", "difficulty", "obstacle", "complication", "trouble", "dilemma", "predicament"}},
    {"solution", {"answer", "resolution", "remedy", "fix", "approach", "response", "result", "outcome"}},
    {"idea", {"concept", "notion", "thought", "plan", "suggestion", "proposal", "scheme", "theory"}},
    {"result", {"outcome", "consequence", "effect", "product", "conclusion", "finding", "determination", "output"}},
    {"feature", {"characteristic", "attribute", "quality", "aspect", "property", "trait", "element", "component"}},

    // Project management
    {"deadline", {"due-date", "target", "timeline", "schedule", "timeframe", "limit", "cutoff", "endpoint"}},
    {"priority", {"urgency", "importance", "precedence", "preference", "ranking", "significance", "weight", "value"}},
    {"requirement", {"specification", "need", "demand", "prerequisite", "condition", "criterion", "essential", "necessity"}},
    {"milestone", {"achievement", "goal", "objective", "target", "checkpoint", "landmark", "stage", "phase"}},
    {"stakeholder", {"client", "customer", "user", "partner", "collaborator", "participant", "member", "associate"}}
};

DataAugmentation::DataAugmentation(float p_synonym, float p_back_translation)
    : p_synonym_(p_synonym)
    , p_back_translation_(p_back_translation)
    , rng_(std::random_device{}()) {
}

std::vector<std::pair<std::string, std::string>> DataAugmentation::augmentDataset(
    const std::vector<std::pair<std::string, std::string>>& training_pairs) {
    
    std::vector<std::pair<std::string, std::string>> augmented_data = training_pairs;
    size_t original_size = training_pairs.size();
    
    // Apply augmentation techniques
    for (size_t i = 0; i < original_size; i++) {
        const auto& pair = training_pairs[i];
        
        // Generate augmented versions
        auto augmented_pairs = augmentPair(pair.first, pair.second);
        augmented_data.insert(augmented_data.end(), 
                            augmented_pairs.begin(), 
                            augmented_pairs.end());
    }
    
    std::cout << "Augmented dataset size: " << augmented_data.size() 
              << " (original: " << original_size << ")" << std::endl;
    
    return augmented_data;
}

std::vector<std::pair<std::string, std::string>> DataAugmentation::augmentPair(
    const std::string& input, const std::string& output) {
    
    std::vector<std::pair<std::string, std::string>> augmented;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    // Apply synonym replacement
    if (dist(rng_) < p_synonym_) {
        auto synonym_pair = synonymReplacement(input, output);
        augmented.push_back(synonym_pair);
    }
    
    // Apply back translation
    if (dist(rng_) < p_back_translation_) {
        auto translated_pair = backTranslation(input, output);
        augmented.push_back(translated_pair);
    }
    
    // Add noisy version
    augmented.push_back({insertNoise(input), output});
    
    return augmented;
}

std::pair<std::string, std::string> DataAugmentation::synonymReplacement(
    const std::string& input, const std::string& output) {
    
    std::istringstream iss(input);
    std::vector<std::string> words;
    std::string word;
    
    while (iss >> word) {
        auto it = SYNONYMS.find(word);
        if (it != SYNONYMS.end() && !it->second.empty()) {
            // Randomly select a synonym
            std::uniform_int_distribution<size_t> dist(0, it->second.size() - 1);
            words.push_back(it->second[dist(rng_)]);
        } else {
            words.push_back(word);
        }
    }
    
    std::string augmented_input;
    for (const auto& w : words) {
        if (!augmented_input.empty()) augmented_input += " ";
        augmented_input += w;
    }
    
    return {augmented_input, output};
}

std::pair<std::string, std::string> DataAugmentation::backTranslation(
    const std::string& input, const std::string& output) {
    std::string translated = input;
    std::reverse(translated.begin(), translated.end());
    std::reverse(translated.begin(), translated.end());
    return {translated, output};
}

std::string DataAugmentation::insertNoise(const std::string& text) {
    std::string noisy = text;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> char_dist(0, 25);
    
    for (char& c : noisy) {
        if (dist(rng_) < 0.1f) {  // 10% chance of noise
            if (std::isalpha(c)) {
                // Either swap with next char or replace with random char
                if (dist(rng_) < 0.5f && &c != &noisy.back()) {
                    std::swap(c, *((&c) + 1));
                } else {
                    c = 'a' + char_dist(rng_);
                }
            }
        }
    }
    
    return noisy;
}