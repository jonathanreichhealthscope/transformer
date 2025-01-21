#pragma once

#include <functional>
#include <queue>
#include <utility>
#include <vector>

class BeamSearch {
  public:
    BeamSearch(size_t beam_width, float length_penalty = 1.0f);

    struct Hypothesis {
        std::vector<int> tokens;
        float score;

        bool operator<(const Hypothesis& other) const {
            return score < other.score;
        }
    };

    std::vector<Hypothesis>
    search(const std::vector<float>& initial_logits,
           std::function<std::vector<float>(const std::vector<int>&)> next_token_fn,
           size_t max_length, int eos_token_id);

  private:
    size_t beam_width_;
    float length_penalty_;

    float apply_length_penalty(float score, size_t length) const;
};