#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

/**
 * @brief Thread-safe logging system for transformer operations.
 * 
 * The Logger class implements a singleton pattern to provide centralized
 * logging functionality throughout the application. Features include:
 * - File and console output
 * - Error level distinction
 * - Timestamp recording
 * - Enable/disable control
 * - Stream redirection
 */
class Logger {
  private:
    std::ofstream log_file;      ///< Output file stream for logging
    std::streambuf* cout_buffer; ///< Stored cout buffer for restoration
    std::streambuf* cerr_buffer; ///< Stored cerr buffer for restoration
    bool logging_enabled;        ///< Whether logging is currently active
    std::string log_file_path;   ///< Path to the current log file

    /// Singleton instance
    static std::unique_ptr<Logger> instance;

    /**
     * @brief Private constructor for singleton pattern.
     * 
     * Initializes logging system in disabled state with
     * no file output.
     */
    Logger();

  public:
    /**
     * @brief Gets the singleton logger instance.
     * @return Reference to the global logger
     */
    static Logger& getInstance();

    /**
     * @brief Starts logging to a file.
     * 
     * Opens the specified file for logging and begins
     * capturing output. Creates directories if needed.
     * 
     * @param file_path Path to log file (default: "transformer.log")
     * @throws std::runtime_error if file cannot be opened
     */
    void startLogging(const std::string& file_path = "transformer.log");

    /**
     * @brief Stops logging and closes the log file.
     * 
     * Restores original stream buffers and closes the
     * log file if open.
     */
    void stopLogging();

    /**
     * @brief Logs a message with optional error level.
     * 
     * Writes a timestamped message to the log file and/or
     * console depending on current settings.
     * 
     * @param message Text to log
     * @param is_error Whether to mark as error (default: false)
     */
    void log(const std::string& message, bool is_error = false);

    /**
     * @brief Checks if logging is currently enabled.
     * @return true if logging is active
     */
    bool isLoggingEnabled() const {
        return logging_enabled;
    }

    /**
     * @brief Temporarily disables logging.
     * 
     * Stops writing to log file but maintains file open.
     */
    void disableLogging();

    /**
     * @brief Re-enables logging after disable.
     * 
     * Resumes writing to log file if one is open.
     */
    void enableLogging() {
        logging_enabled = true;
    }

    // Prevent copying and assignment for singleton
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    /**
     * @brief Destructor that ensures proper cleanup.
     * 
     * Stops logging and closes files before destruction.
     */
    ~Logger();
};

#endif // LOGGER_HPP