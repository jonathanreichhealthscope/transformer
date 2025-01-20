#include "beam_search.hpp"
#include <cmath>
#include <algorithm>

BeamSearch::BeamSearch(size_t beam_width, float length_penalty)
    : beam_width_(beam_width), length_penalty_(length_penalty) {}

float BeamSearch::apply_length_penalty(float score, size_t length) const {
    return score / std::pow(length, length_penalty_);
}

std::vector<BeamSearch::Hypothesis> BeamSearch::search(
    const std::vector<float>& initial_logits,
    std::function<std::vector<float>(const std::vector<int>&)> next_token_fn,
    size_t max_length,
    int eos_token_id
) {
    std::priority_queue<Hypothesis> active_beams;
    std::vector<Hypothesis> finished_beams;
    
    // Initialize with top-k tokens from initial distribution
    std::vector<std::pair<float, int>> initial_topk;
    for (size_t i = 0; i < initial_logits.size(); ++i) {
        initial_topk.emplace_back(initial_logits[i], i);
    }
    
    std::partial_sort(
        initial_topk.begin(),
        initial_topk.begin() + beam_width_,
        initial_topk.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );
    
    for (size_t i = 0; i < beam_width_; ++i) {
        Hypothesis h{{initial_topk[i].second}, initial_topk[i].first};
        active_beams.push(h);
    }
    
    // Main beam search loop
    while (!active_beams.empty() && active_beams.top().tokens.size() < max_length) {
        std::vector<Hypothesis> current_beams;
        size_t num_active = active_beams.size();
        
        for (size_t i = 0; i < num_active; ++i) {
            auto current = active_beams.top();
            active_beams.pop();
            
            // Get next token distribution
            auto next_logits = next_token_fn(current.tokens);
            
            // Find top-k next tokens
            std::vector<std::pair<float, int>> topk;
            for (size_t j = 0; j < next_logits.size(); ++j) {
                topk.emplace_back(next_logits[j] + current.score, j);
            }
            
            std::partial_sort(
                topk.begin(),
                topk.begin() + beam_width_,
                topk.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; }
            );
            
            // Create new hypotheses
            for (size_t j = 0; j < beam_width_; ++j) {
                Hypothesis new_h = current;
                new_h.tokens.push_back(topk[j].second);
                new_h.score = apply_length_penalty(topk[j].first, new_h.tokens.size());
                
                if (topk[j].second == eos_token_id) {
                    finished_beams.push_back(new_h);
                } else {
                    current_beams.push_back(new_h);
                }
            }
        }
        
        // Keep top-k beams for next iteration
        active_beams = std::priority_queue<Hypothesis>();
        std::partial_sort(
            current_beams.begin(),
            current_beams.begin() + std::min(beam_width_, current_beams.size()),
            current_beams.end()
        );
        
        for (size_t i = 0; i < std::min(beam_width_, current_beams.size()); ++i) {
            active_beams.push(current_beams[i]);
        }
    }
    
    // Add any remaining active beams to finished beams
    while (!active_beams.empty()) {
        finished_beams.push_back(active_beams.top());
        active_beams.pop();
    }
    
    // Sort finished beams by score
    std::sort(finished_beams.begin(), finished_beams.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });
    
    return finished_beams;
} 