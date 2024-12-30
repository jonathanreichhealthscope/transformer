#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <fstream>
#include <string>
#include <ctime>
#include <iostream>
#include <memory>

class Logger {
private:
    std::ofstream log_file;
    std::streambuf* cout_buffer;
    std::streambuf* cerr_buffer;

    static std::unique_ptr<Logger> instance;
    Logger(); // Private constructor for singleton

public:
    static Logger& getInstance();
    
    void startLogging();
    void stopLogging();
    void log(const std::string& message, bool is_error = false);
    
    // Prevent copying and assignment
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    ~Logger();
};

#endif // LOGGER_HPP 