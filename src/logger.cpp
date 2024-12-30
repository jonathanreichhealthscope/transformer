#include "../include/logger.hpp"

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

Logger& Logger::getInstance() {
    if (!instance) {
        instance = std::unique_ptr<Logger>(new Logger());
    }
    return *instance;
}

void Logger::startLogging() {
    // Store the current buffers
    cout_buffer = std::cout.rdbuf();
    cerr_buffer = std::cerr.rdbuf();
    
    // Redirect stdout and stderr to the log file
    std::cout.rdbuf(log_file.rdbuf());
    std::cerr.rdbuf(log_file.rdbuf());
    
    // Write initial log entry with timestamp
    time_t now = time(nullptr);
    log_file << "\n=== Logging started at " << ctime(&now) << "===\n";
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

void Logger::log(const std::string& message, bool is_error) {
    time_t now = time(nullptr);
    std::string timestamp(ctime(&now));
    timestamp = timestamp.substr(0, timestamp.length() - 1); // Remove newline
    
    log_file << "[" << timestamp << "] " 
             << (is_error ? "ERROR: " : "INFO: ") 
             << message << std::endl;
} 