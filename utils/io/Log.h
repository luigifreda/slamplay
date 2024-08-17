#pragma once

#include <cstring>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>

#include <chrono>
#include <ctime>
#include <iomanip>

#include "io/LogColors.h"

namespace slamplay {

/// modified from the base class of http://drdobbs.com/cpp/201804215

enum class LogLevel { error = 0,
                      warn = 1,
                      info = 2,
                      debug = 3,
                      debug1 = 4,
                      debug2 = 5,
                      debug3 = 6,
                      debug4 = 7 };
static const std::string LogLevelStrings[] = {"ERROR", "WARNING", "INFO", "DEBUG", "DEBUG1", "DEBUG2", "DEBUG3", "DEBUG4"};
static const std::string LogLevelColors[] = {LOG_COL_RED, LOG_COL_BROWN, LOG_COL_GREEN, LOG_COL_CYAN, LOG_COL_CYAN, LOG_COL_CYAN, LOG_COL_CYAN, LOG_COL_CYAN};

#ifndef LogError
#define LogError Log().Get(LogLevel::error)
#endif
#ifndef LogWarning
#define LogWarning Log().Get(LogLevel::warn)
#endif
#ifndef LogInfo
#define LogInfo Log().Get(LogLevel::info)
#endif
#ifndef LogDebug
#define LogDebug Log().Get(LogLevel::debug)
#endif

///	\class Log
///	\author Luigi Freda
///	\brief A class implementing a standard log with different reporting levels
class Log {
   public:
    Log() : _messageLevel(LogLevel::debug) {};

    inline virtual ~Log();

    inline std::ostringstream& Get(LogLevel level = LogLevel::error);
    inline std::ostringstream& Get(int level);

    /// enable/disable report with time
    static void SetWithTime(bool val) { _bWithTime = val; }

   public:
    static const int maxReportingLevel = 7;
    inline static LogLevel& ReportingLevel() { return _reportingLevel; }
    static LogLevel _reportingLevel;

   protected:
    std::ostringstream _os;

   private:
    Log(const Log&);
    Log& operator=(const Log&);

   private:
    static bool _bWithTime;
    LogLevel _messageLevel;
};

LogLevel Log::_reportingLevel = LogLevel::error;
bool Log::_bWithTime = false;

std::ostringstream& Log::Get(const LogLevel level) {
    return Get(static_cast<int>(level));
}

std::ostringstream& Log::Get(int level) {
    level = std::min(level, (int)Log::maxReportingLevel);
    _os << LogLevelColors[level] << "[" << LogLevelStrings[level] << "]: ";
    if (level > static_cast<int>(LogLevel::debug))
        _os << std::string(level - static_cast<int>(LogLevel::debug), '\t');
    _messageLevel = (LogLevel)level;
    return _os;
}

Log::~Log() {
    if (_messageLevel >= Log::ReportingLevel())
    {
        if (_bWithTime)
        {
            /// reporting time with milliseconds precision
            auto now = std::chrono::system_clock::now();
            auto now_c = std::chrono::system_clock::to_time_t(now);
            auto now_tm = std::localtime(&now_c);

            auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                              now.time_since_epoch())
                              .count() %
                          1000;
            std::ostringstream ss_millis;
            ss_millis << std::setw(3) << std::setfill('0') << millis;

            std::cout << "[" << std::put_time(now_tm, "%T") << ":" << ss_millis.str() << "]";
        }
        _os << LOG_COL_NORMAL;
        _os << std::endl;
        std::cout << _os.str();
    }
}

}  // namespace slamplay
