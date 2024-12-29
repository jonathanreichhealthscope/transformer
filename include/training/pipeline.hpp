#pragma once
#include "../transformer.hpp"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

class PipelinedTrainer {
private:
  struct Stage {
    std::queue<Matrix> input_queue;
    std::queue<Matrix> output_queue;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> should_stop{false};
    std::function<Matrix(const Matrix &)> process_fn;
  };

  std::vector<Stage> pipeline_stages;
  size_t num_micro_batches;
  size_t batch_size;
  std::atomic<bool> training_active{false};

  Matrix wait_for_input(size_t stage_idx) {
    auto &stage = pipeline_stages[stage_idx];
    std::unique_lock<std::mutex> lock(stage.mutex);
    stage.cv.wait(lock, [&]() {
      return !stage.input_queue.empty() || stage.should_stop;
    });

    if (stage.should_stop && stage.input_queue.empty()) {
      return Matrix();
    }

    Matrix input = std::move(stage.input_queue.front());
    stage.input_queue.pop();
    return input;
  }

  void send_to_next_stage(size_t stage_idx, Matrix output) {
    if (stage_idx + 1 < pipeline_stages.size()) {
      auto &next_stage = pipeline_stages[stage_idx + 1];
      {
        std::lock_guard<std::mutex> lock(next_stage.mutex);
        next_stage.input_queue.push(std::move(output));
      }
      next_stage.cv.notify_one();
    }
  }

public:
  PipelinedTrainer(size_t num_stages, size_t micro_batches, size_t batch_size_)
      : num_micro_batches(micro_batches), batch_size(batch_size_) {
    pipeline_stages.resize(num_stages);
  }

  void set_stage_function(size_t stage_idx,
                          std::function<Matrix(const Matrix &)> fn) {
    if (stage_idx < pipeline_stages.size()) {
      pipeline_stages[stage_idx].process_fn = std::move(fn);
    }
  }

  void start() {
    training_active = true;
    schedule_forward();
  }

  void stop() {
    training_active = false;
    for (auto &stage : pipeline_stages) {
      stage.should_stop = true;
      stage.cv.notify_all();
      if (stage.worker.joinable()) {
        stage.worker.join();
      }
    }
  }

  void feed_input(const Matrix &input) {
    if (!pipeline_stages.empty()) {
      auto &first_stage = pipeline_stages[0];
      {
        std::lock_guard<std::mutex> lock(first_stage.mutex);
        first_stage.input_queue.push(input);
      }
      first_stage.cv.notify_one();
    }
  }

  std::optional<Matrix> get_output() {
    if (pipeline_stages.empty())
      return std::nullopt;

    auto &last_stage = pipeline_stages.back();
    std::unique_lock<std::mutex> lock(last_stage.mutex);
    if (last_stage.output_queue.empty())
      return std::nullopt;

    Matrix output = std::move(last_stage.output_queue.front());
    last_stage.output_queue.pop();
    return output;
  }

  ~PipelinedTrainer() { stop(); }
};