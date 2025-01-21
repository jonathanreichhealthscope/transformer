#include "../include/logger.hpp"
#include <chrono>
#include <filesystem>

std::unique_ptr<Logger> Logger::instance = nullptr;

Logger::Logger() : logging_enabled(false), cout_buffer(nullptr), cerr_buffer(nullptr) {
    // Initialize in constructor
}

Logger::~Logger() {
    if (logging_enabled) {
        stopLogging();
    }
}

Logger& Logger::getInstance() {
    if (!instance) {
        instance = std::unique_ptr<Logger>(new Logger());
    }
    return *instance;
}

void Logger::startLogging(const std::string& file_path) {
    if (logging_enabled) {
        stopLogging();
    }

    log_file_path = file_path;

    // Create directories if they don't exist
    std::filesystem::path path(file_path);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    // Open log file with immediate flush on write
    log_file.open(file_path, std::ios::out | std::ios::trunc);
    if (!log_file.is_open()) {
        std::cerr << "Failed to open log file: " << file_path << std::endl;
        return;
    }

    // Set log file to auto-flush
    log_file.rdbuf()->pubsetbuf(nullptr, 0);

    // Store and redirect cout buffer
    cout_buffer = std::cout.rdbuf();
    std::cout.rdbuf(log_file.rdbuf());

    // Store and redirect cerr buffer
    cerr_buffer = std::cerr.rdbuf();
    std::cerr.rdbuf(log_file.rdbuf());

    // Enable auto-flush for cout and cerr
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    logging_enabled = true;

    // Log start time
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    log_file << "=== Logging started at " << std::ctime(&time) << "===" << std::endl;
    log_file.flush();
}

void Logger::stopLogging() {
    if (!logging_enabled) {
        return;
    }

    // Ensure all buffers are flushed
    std::cout.flush();
    std::cerr.flush();
    log_file.flush();

    // Log stop time
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    log_file << "\n=== Logging stopped at " << std::ctime(&time) << "===" << std::endl;

    // Restore cout buffer and its flags
    if (cout_buffer) {
        std::cout.rdbuf(cout_buffer);
        std::cout.unsetf(std::ios::unitbuf);
        cout_buffer = nullptr;
    }

    // Restore cerr buffer and its flags
    if (cerr_buffer) {
        std::cerr.rdbuf(cerr_buffer);
        std::cerr.unsetf(std::ios::unitbuf);
        cerr_buffer = nullptr;
    }

    // Close log file
    if (log_file.is_open()) {
        log_file.close();
    }

    logging_enabled = false;
}

void Logger::disableLogging() {
    if (logging_enabled) {
        stopLogging();
    }
    logging_enabled = false;
}

void Logger::log(const std::string& message, bool is_error) {
    if (!logging_enabled) {
        return;
    }

    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::string timestamp(std::ctime(&time));
    timestamp = timestamp.substr(0, timestamp.length() - 1); // Remove trailing newline

    log_file << "[" << timestamp << "] " << (is_error ? "ERROR: " : "INFO: ") << message
             << std::endl;
    log_file.flush();
}