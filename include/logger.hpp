#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

class Logger {
private:
  std::ofstream log_file;
  std::streambuf *cout_buffer;
  std::streambuf *cerr_buffer;
  bool logging_enabled = false;

  static std::unique_ptr<Logger> instance;
  Logger(); // Private constructor for singleton

public:
  static Logger &getInstance();

  void startLogging();
  void stopLogging();
  void log(const std::string &message, bool is_error = false);
  void disableLogging() { logging_enabled = false; }

  // Prevent copying and assignment
  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  ~Logger();
};

#endif // LOGGER_HPP