// *************************************************************************
/* 
 * This file is part of the slamplay project.
 * Copyright (C) 2018-present Luigi Freda <luigifreda at gmail dot com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version, at your option. If this file is a modified/adapted 
 * version of an original file distributed under a different license that 
 * is not compatible with the GNU General Public License, the 
 * BSD 3-Clause License will apply instead.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
// *************************************************************************
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
