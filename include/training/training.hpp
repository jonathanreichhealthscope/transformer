#pragma once

// Include all training components
#include "loss_tracker.hpp"
#include "gradient_manager.hpp"
#include "learning_rate_scheduler.hpp"
#include "training_state_manager.hpp"
#include "training_monitor.hpp"

// Forward declare TrainingMetrics to resolve circular dependency
struct TrainingMetrics;

// Re-export all training components
using TrainingStateManagerPtr = std::unique_ptr<TrainingStateManager>;
using TrainingMonitorPtr = std::unique_ptr<TrainingMonitor>; 