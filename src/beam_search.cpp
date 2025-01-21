#include "beam_search.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

BeamSearch::BeamSearch(size_t beam_width, float length_penalty)
    : beam_width_(beam_width), length_penalty_(length_penalty) {
    std::cout << "Initializing BeamSearch with width=" << beam_width
              << ", length_penalty=" << length_penalty << std::endl;
}

float BeamSearch::apply_length_penalty(float score, size_t length) const {
    float penalized_score = score / std::pow(length, length_penalty_);
    std::cout << "Applied length penalty: original_score=" << std::fixed << std::setprecision(4)
              << score << ", length=" << length << ", penalized_score=" << penalized_score
              << std::endl;
    return penalized_score;
}

std::vector<BeamSearch::Hypothesis>
BeamSearch::search(const std::vector<float>& initial_logits,
                   std::function<std::vector<float>(const std::vector<int>&)> next_token_fn,
                   size_t max_length, int eos_token_id) {
    std::cout << "\n=== Starting Beam Search ===\n"
              << "Max length: " << max_length << "\n"
              << "EOS token ID: " << eos_token_id << "\n"
              << "Initial logits size: " << initial_logits.size() << std::endl;

    std::priority_queue<Hypothesis> active_beams;
    std::vector<Hypothesis> finished_beams;

    // Initialize with top-k tokens
    std::vector<std::pair<float, int>> initial_topk;
    for (size_t i = 0; i < initial_logits.size(); ++i) {
        initial_topk.emplace_back(initial_logits[i], i);
    }

    std::partial_sort(initial_topk.begin(), initial_topk.begin() + beam_width_, initial_topk.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    std::cout << "\nInitial top-" << beam_width_ << " tokens:" << std::endl;
    for (size_t i = 0; i < beam_width_; ++i) {
        std::cout << "Token " << initial_topk[i].second << " (score: " << std::fixed
                  << std::setprecision(4) << initial_topk[i].first << ")" << std::endl;

        Hypothesis h{{initial_topk[i].second}, initial_topk[i].first};
        active_beams.push(h);
    }

    // Main beam search loop
    size_t step = 0;
    while (!active_beams.empty() && active_beams.top().tokens.size() < max_length) {
        step++;
        std::cout << "\n=== Beam Search Step " << step << " ===\n"
                  << "Active beams: " << active_beams.size() << "\n"
                  << "Finished beams: " << finished_beams.size() << std::endl;

        std::vector<Hypothesis> current_beams;
        size_t num_active = active_beams.size();

        for (size_t i = 0; i < num_active; ++i) {
            auto current = active_beams.top();
            active_beams.pop();

            std::cout << "\nProcessing beam " << i + 1 << "/" << num_active
                      << " (score: " << current.score << ")" << std::endl;

            // Get next token distribution
            auto next_logits = next_token_fn(current.tokens);

            // Find top-k next tokens
            std::vector<std::pair<float, int>> topk;
            for (size_t j = 0; j < next_logits.size(); ++j) {
                topk.emplace_back(next_logits[j] + current.score, j);
            }

            std::partial_sort(topk.begin(), topk.begin() + beam_width_, topk.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });

            std::cout << "Top " << beam_width_ << " next tokens:" << std::endl;
            // Create new hypotheses
            for (size_t j = 0; j < beam_width_; ++j) {
                Hypothesis new_h = current;
                new_h.tokens.push_back(topk[j].second);
                new_h.score = apply_length_penalty(topk[j].first, new_h.tokens.size());

                std::cout << "  Token " << topk[j].second << " (score: " << std::fixed
                          << std::setprecision(4) << new_h.score << ")";

                if (topk[j].second == eos_token_id) {
                    finished_beams.push_back(new_h);
                    std::cout << " -> Finished";
                } else {
                    current_beams.push_back(new_h);
                    std::cout << " -> Active";
                }
                std::cout << std::endl;
            }
        }

        // Keep top-k beams for next iteration
        active_beams = std::priority_queue<Hypothesis>();
        std::partial_sort(current_beams.begin(),
                          current_beams.begin() + std::min(beam_width_, current_beams.size()),
                          current_beams.end());

        std::cout << "\nSelected beams for next iteration:" << std::endl;
        for (size_t i = 0; i < std::min(beam_width_, current_beams.size()); ++i) {
            std::cout << "Beam " << i + 1 << " (score: " << current_beams[i].score
                      << ", length: " << current_beams[i].tokens.size() << ")" << std::endl;
            active_beams.push(current_beams[i]);
        }
    }

    // Add remaining active beams to finished beams
    std::cout << "\nAdding remaining active beams to finished beams..." << std::endl;
    while (!active_beams.empty()) {
        finished_beams.push_back(active_beams.top());
        std::cout << "Added beam with score: " << active_beams.top().score << std::endl;
        active_beams.pop();
    }

    // Sort finished beams by score
    std::sort(finished_beams.begin(), finished_beams.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });

    std::cout << "\n=== Beam Search Complete ===\n"
              << "Total finished beams: " << finished_beams.size() << "\n"
              << "Top 3 beam scores:" << std::endl;

    for (size_t i = 0; i < std::min(size_t(3), finished_beams.size()); ++i) {
        std::cout << i + 1 << ". Score: " << finished_beams[i].score
                  << " (length: " << finished_beams[i].tokens.size() << ")" << std::endl;
    }

    return finished_beams;
}