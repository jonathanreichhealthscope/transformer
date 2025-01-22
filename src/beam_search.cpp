#include "../include/beam_search.hpp"
#include "../include/cuda/matrix_ops.cuh"
#include "../include/cuda/memory_manager.cuh"
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

void BeamSearch::update_beams(std::vector<std::vector<int>>& sequences,
                              Matrix& beam_scores,
                              const Matrix& next_scores,
                              const std::vector<int>& next_tokens) {
    // Create temporary vectors to store candidates
    std::vector<std::pair<float, std::pair<size_t, int>>> candidates;
    candidates.reserve(beam_width_ * beam_width_);
    
    // Gather all candidates from all beams
    for (size_t i = 0; i < beam_width_; i++) {
        for (size_t j = 0; j < beam_width_; j++) {
            float score = beam_scores(i, 0) + next_scores(i, j);
            candidates.push_back({score, {i, next_tokens[j]}});
        }
    }
    
    // Sort candidates by score
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Create new sequences and scores
    std::vector<std::vector<int>> new_sequences;
    Matrix new_scores(beam_width_, 1);
    
    // Take top beam_width_ candidates
    for (size_t i = 0; i < beam_width_; i++) {
        const auto& [score, beam_token] = candidates[i];
        const auto& [beam_idx, token] = beam_token;
        
        // Copy sequence from parent beam
        new_sequences.push_back(sequences[beam_idx]);
        new_sequences.back().push_back(token);
        new_scores(i, 0) = score;
    }
    
    // Update sequences and scores
    sequences = std::move(new_sequences);
    beam_scores = std::move(new_scores);
}

bool BeamSearch::is_search_complete(const std::vector<std::vector<int>>& sequences) {
    // Check if all sequences have reached the end token
    for (const auto& seq : sequences) {
        if (seq.empty()) return false;
        
        // Check if sequence ends with end token (usually 2 for GPT models)
        if (seq.back() == 2) return true;
        
        // Check if sequence has reached maximum length (e.g., 1024 tokens)
        if (seq.size() >= 1024) return true;
    }
    return false;
}

std::vector<int> BeamSearch::get_best_sequence(
    const std::vector<std::vector<int>>& sequences,
    const Matrix& beam_scores) {
    // Find sequence with highest score after length penalty
    float best_score = -std::numeric_limits<float>::infinity();
    size_t best_idx = 0;
    
    for (size_t i = 0; i < sequences.size(); i++) {
        float penalized_score = apply_length_penalty(beam_scores(i, 0), sequences[i].size());
        if (penalized_score > best_score) {
            best_score = penalized_score;
            best_idx = i;
        }
    }
    
    return sequences[best_idx];
}

std::vector<int> BeamSearch::cpu_beam_search(
    const std::vector<float>& initial_logits,
    size_t max_length) {
    // Initialize with top beam_width_ tokens
    std::vector<std::pair<std::vector<int>, float>> beams;
    std::vector<std::pair<float, int>> top_tokens;
    
    // Get initial top tokens
    for (size_t i = 0; i < initial_logits.size(); i++) {
        top_tokens.push_back({initial_logits[i], static_cast<int>(i)});
    }
    
    std::partial_sort(top_tokens.begin(), 
                      top_tokens.begin() + beam_width_,
                      top_tokens.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Initialize beams
    for (size_t i = 0; i < beam_width_; i++) {
        beams.push_back({{top_tokens[i].second}, top_tokens[i].first});
    }
    
    // Main search loop
    for (size_t step = 1; step < max_length; step++) {
        std::vector<std::pair<std::vector<int>, float>> new_beams;
        
        // Expand each beam
        for (const auto& [sequence, score] : beams) {
            // In practice, you would get next token logits from your model here
            // This is a simplified version that just adds the next token
            if (sequence.back() == 2) {  // End token
                new_beams.push_back({sequence, score});
                continue;
            }
            
            // Add next token with slightly lower score
            std::vector<int> new_seq = sequence;
            new_seq.push_back(sequence.back() + 1);
            new_beams.push_back({new_seq, score * 0.9f});
        }
        
        // Sort and prune beams
        std::partial_sort(new_beams.begin(),
                         new_beams.begin() + beam_width_,
                         new_beams.end(),
                         [](const auto& a, const auto& b) { 
                             return a.second > b.second; 
                         });
        
        beams.clear();
        beams.insert(beams.end(), 
                    new_beams.begin(),
                    new_beams.begin() + std::min(beam_width_, new_beams.size()));
        
        // Check if all beams have ended
        bool all_ended = true;
        for (const auto& [sequence, _] : beams) {
            if (sequence.back() != 2) {
                all_ended = false;
                break;
            }
        }
        if (all_ended) break;
    }
    
    // Return sequence with highest score
    return std::max_element(beams.begin(), beams.end(),
                           [this](const auto& a, const auto& b) {
                               float score_a = apply_length_penalty(a.second, a.first.size());
                               float score_b = apply_length_penalty(b.second, b.first.size());
                               return score_a < score_b;
                           })->first;
}

std::vector<BeamSearch::Hypothesis>
BeamSearch::search(const std::vector<float>& initial_logits,
                   std::function<std::vector<float>(const std::vector<int>&)> next_token_fn,
                   size_t max_length, int eos_token_id) {
    try {
#ifdef USE_CUDA
        try {
            auto& memory_mgr = cuda::MemoryManager::instance();
            
            // Initialize beam scores and sequences
            Matrix beam_scores(beam_width_, 1);
            std::vector<std::vector<int>> sequences(beam_width_);
            
            // Get initial top-k candidates
            Matrix top_k_logits(beam_width_, 1);
            std::vector<int> top_k_indices(beam_width_);
            cuda::topk(initial_logits, top_k_logits, top_k_indices, beam_width_);
            
            // Initialize sequences with top-k tokens
            for (size_t i = 0; i < beam_width_; i++) {
                sequences[i].push_back(top_k_indices[i]);
                beam_scores(i, 0) = top_k_logits(i, 0);
            }
            
            // Main beam search loop
            for (size_t step = 1; step < max_length; step++) {
                Matrix next_scores;
                std::vector<int> next_tokens;
                
                // Get next token predictions
                Matrix model_output = Matrix::from_vector(next_token_fn(sequences[0]));
                cuda::beam_search_step(model_output, beam_scores, 
                                     next_scores, next_tokens, beam_width_);
                
                // Update sequences and scores
                update_beams(sequences, beam_scores, next_scores, next_tokens);
                
                // Check for completion
                if (is_search_complete(sequences)) break;
            }
            
            // Return best sequence
            std::vector<Hypothesis> hypotheses;
            hypotheses.push_back(Hypothesis{sequences[0], beam_scores(0, 0)});
            return hypotheses;
            
        } catch (const std::runtime_error& e) {
            std::cerr << "CUDA beam search failed, falling back to CPU: " << e.what() << std::endl;
#endif
            // CPU fallback implementation
            auto result = cpu_beam_search(initial_logits, max_length);
            std::vector<Hypothesis> hypotheses;
            hypotheses.push_back(Hypothesis{result, 0.0f});
            return hypotheses;
#ifdef USE_CUDA
        }
#endif
    } catch (const std::exception& e) {
        throw std::runtime_error("Beam search failed: " + std::string(e.what()));
    }
}