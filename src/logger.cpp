#include "../include/logger.hpp"
#include <chrono>

std::unique_ptr<Logger> Logger::instance = nullptr;

Logger::Logger() {
  log_file.open("transformer_log.log", std::ios::out | std::ios::app);
  cout_buffer = nullptr;
  cerr_buffer = nullptr;
}

Logger::~Logger() {
  if (log_file.is_open()) {
    stopLogging();
    log_file.close();
  }
}

Logger &Logger::getInstance() {
  if (!instance) {
    instance = std::unique_ptr<Logger>(new Logger());
  }
  return *instance;
}

void Logger::startLogging() {
  // Open the file in truncation mode to clear previous contents
  log_file.open("transformer_log.log", std::ios::out | std::ios::trunc);
  if (!log_file.is_open()) {
    std::cerr << "Failed to open log file" << std::endl;
    return;
  }
  
  // Start capturing cout
  auto old_cout_buf = std::cout.rdbuf();
  std::cout.rdbuf(log_file.rdbuf());
  
  // Log start time
  auto now = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(now);
  log_file << "=== Logging started at " << std::ctime(&time) << "===" << std::endl;
}

void Logger::stopLogging() {
  // Restore the original buffers
  if (cout_buffer) {
    std::cout.rdbuf(cout_buffer);
    cout_buffer = nullptr;
  }
  if (cerr_buffer) {
    std::cerr.rdbuf(cerr_buffer);
    cerr_buffer = nullptr;
  }

  time_t now = time(nullptr);
  log_file << "\n=== Logging stopped at " << ctime(&now) << "===\n";
}

void Logger::log(const std::string &message, bool is_error) {
  time_t now = time(nullptr);
  std::string timestamp(ctime(&now));
  timestamp = timestamp.substr(0, timestamp.length() - 1); // Remove newline

  log_file << "[" << timestamp << "] " << (is_error ? "ERROR: " : "INFO: ")
           << message << std::endl;
}