#include "../include/beam_search.hpp"
#include "../include/cuda/matrix_ops.cuh"
#include "../include/cuda/memory_manager.cuh"
#include "../include/utils.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

BeamSearch::BeamSearch(size_t beam_width, float length_penalty, float temperature,
                       float diversity_strength, size_t top_k, float top_p)
    : beam_width_(beam_width)
    , length_penalty_(length_penalty)
    , temperature(temperature)
    , diversity_strength(diversity_strength)
    , top_k(top_k)
    , top_p(top_p) {
    std::cout << "Initializing BeamSearch with width=" << beam_width
              << ", length_penalty=" << length_penalty 
              << ", temperature=" << temperature
              << ", diversity_strength=" << diversity_strength
              << ", top_k=" << top_k
              << ", top_p=" << top_p << std::endl;
    // Use consistent special token IDs
    pad_token_id_ = 0;
    unk_token_id_ = 1;
    bos_token_id_ = 2;
    eos_token_id_ = 3;
    mask_token_id_ = 4;
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
    std::function<std::vector<float>(const std::vector<int>&)> next_token_fn,
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
            if (sequence.back() == 2) {  // End token
                new_beams.push_back({sequence, score});
                continue;
            }
            
            // Get next token logits from the model
            std::vector<float> next_logits = next_token_fn(sequence);
            
            // Apply temperature scaling and sampling
            std::vector<float> scaled_probs = calculateScores(next_logits);
            
            // Get top-k candidates
            auto candidate_pairs = topKSampling(scaled_probs, top_k);
            std::vector<size_t> candidate_indices;
            for (const auto& pair : candidate_pairs) {
                candidate_indices.push_back(pair.second);
            }
            
            // Apply nucleus sampling if enabled
            if (top_p < 1.0f) {
                auto nucleus_pairs = nucleusSampling(scaled_probs, top_p);
                std::vector<size_t> nucleus_indices;
                for (const auto& pair : nucleus_pairs) {
                    nucleus_indices.push_back(pair.second);
                }
                
                // Use intersection of top-k and nucleus sampling
                std::vector<size_t> filtered_indices;
                std::set_intersection(
                    candidate_indices.begin(), candidate_indices.end(),
                    nucleus_indices.begin(), nucleus_indices.end(),
                    std::back_inserter(filtered_indices)
                );
                candidate_indices = filtered_indices;
            }
            
            // Create beam candidates
            std::vector<BeamCandidate> candidates;
            for (size_t idx : candidate_indices) {
                // Convert int sequence to size_t sequence
                std::vector<size_t> new_sequence;
                new_sequence.reserve(sequence.size() + 1);
                for (int token : sequence) {
                    new_sequence.push_back(static_cast<size_t>(token));
                }
                new_sequence.push_back(idx);
                float new_score = score + std::log(scaled_probs[idx]);
                candidates.push_back(BeamCandidate(new_sequence, new_score));
            }
            
            // Apply diversity penalty
            diversityPenalty(candidates, diversity_strength);
            
            // Add top candidates to new beams
            std::sort(candidates.begin(), candidates.end(),
                     [](const auto& a, const auto& b) { return a.score > b.score; });
            
            size_t num_to_add = std::min(beam_width_, candidates.size());
            for (size_t i = 0; i < num_to_add; i++) {
                // Convert size_t sequence back to int sequence
                std::vector<int> int_sequence;
                int_sequence.reserve(candidates[i].sequence.size());
                for (size_t token : candidates[i].sequence) {
                    int_sequence.push_back(static_cast<int>(token));
                }
                new_beams.push_back({int_sequence, candidates[i].score});
            }
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
    
    // Return sequence with highest score after length penalty
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
            auto result = cpu_beam_search(initial_logits, next_token_fn, max_length);
            std::vector<Hypothesis> hypotheses;
            hypotheses.push_back(Hypothesis(result, 0.0f));
            return hypotheses;
#ifdef USE_CUDA
        }
#endif
    } catch (const std::exception& e) {
        throw std::runtime_error("Beam search failed: " + std::string(e.what()));
    }
}

std::vector<float> BeamSearch::calculateScores(const std::vector<float>& logits) {
    // Increase temperature for more randomness with small datasets
    float higher_temp = 1.2f;  // Up from 0.8f
    std::vector<float> scaled_logits(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        scaled_logits[i] = logits[i] / higher_temp;
    }
    
    // Apply softmax on temperature-scaled logits
    float max_logit = *std::max_element(scaled_logits.begin(), scaled_logits.end());
    float sum_exp = 0.0f;
    std::vector<float> probs(scaled_logits.size());
    
    for (size_t i = 0; i < scaled_logits.size(); i++) {
        probs[i] = std::exp(scaled_logits[i] - max_logit);
        sum_exp += probs[i];
    }
    
    for (float& prob : probs) {
        prob /= sum_exp;
    }
    
    return probs;
}

// Increase diversity penalty
void BeamSearch::diversityPenalty(std::vector<BeamCandidate>& candidates, float strength) {
    // Use stronger diversity penalty for small datasets
    float stronger_penalty = strength * 1.5f;  // Increase the penalty
    for (size_t i = 0; i < candidates.size(); i++) {
        for (size_t j = 0; j < i; j++) {
            float overlap = calculateOverlap(candidates[i].sequence, candidates[j].sequence);
            candidates[i].score -= stronger_penalty * overlap;
        }
    }
}

std::vector<std::pair<float, size_t>> BeamSearch::topKSampling(
    const std::vector<float>& probabilities, size_t k) {
    std::vector<std::pair<float, size_t>> prob_idx;
    prob_idx.reserve(probabilities.size());
    
    for (size_t i = 0; i < probabilities.size(); i++) {
        prob_idx.push_back({probabilities[i], i});
    }
    
    // Sort by probability in descending order
    std::partial_sort(
        prob_idx.begin(),
        prob_idx.begin() + std::min(k, prob_idx.size()),
        prob_idx.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );
    
    // Return top k pairs
    return std::vector<std::pair<float, size_t>>(
        prob_idx.begin(),
        prob_idx.begin() + std::min(k, prob_idx.size())
    );
}

std::vector<std::pair<float, size_t>> BeamSearch::nucleusSampling(
    const std::vector<float>& probabilities, float p) {
    std::vector<std::pair<float, size_t>> sorted_probs;
    sorted_probs.reserve(probabilities.size());
    
    for (size_t i = 0; i < probabilities.size(); i++) {
        sorted_probs.push_back({probabilities[i], i});
    }
    
    // Sort by probability in descending order
    std::sort(sorted_probs.begin(), sorted_probs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Find cutoff for cumulative probability >= p
    float cumsum = 0.0f;
    size_t cutoff_idx = sorted_probs.size();
    
    for (size_t i = 0; i < sorted_probs.size(); i++) {
        cumsum += sorted_probs[i].first;
        if (cumsum >= p) {
            cutoff_idx = i + 1;
            break;
        }
    }
    
    return std::vector<std::pair<float, size_t>>(
        sorted_probs.begin(),
        sorted_probs.begin() + cutoff_idx
    );
}