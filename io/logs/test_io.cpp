#include <math.h>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "io/Log.h"
#include "io/Logger.h"

using namespace slamplay;
using namespace std::chrono_literals;

int main(int argc, char **argv) {
    /// Logger class test
    Logger logger("INFO", LOG_COL_GREEN);
    logger << "prova " << 1 << std::endl;

    LoggerFile loggerFile("info.txt");
    loggerFile << "prova " << 1 << std::endl;
    loggerFile << "prova2 " << 1 << std::endl;

    std::cout << "--- " << std::endl;

    /// Log class test
    Log::ReportingLevel() = LogLevel::error;
    const int count = 10;

    Log().Get(LogLevel::info) << "A loop with " << count << " iterations";
    Log().Get(LogLevel::warn) << "A loop with " << count << " iterations";
    Log().Get(LogLevel::debug) << "A loop with " << count << " iterations";
    for (int i = 0; i < count; ++i)
    {
        Log().Get(i) << "the counter i = " << i;
    }

    Log::SetWithTime(true);
    for (int i = 0; i < 10; ++i)
    {
        Log().Get(LogLevel::error) << "counter i = " << i;
        std::this_thread::sleep_for(100ms);  // can write this thanks to std::chrono_literals
    }
}
