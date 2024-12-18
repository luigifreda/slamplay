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
#include <unistd.h>
#include <iostream>
#include <thread>

#include "time/ChronoTimer.h"

using namespace std::chrono_literals;

int main(int argc, char **argv) {
    slamplay::ChronoTimer timer;

    int N = 1000;
    auto sleepTime = 10ms;  // can write this thanks to std::chrono_literals;

    for (int i = 0; i < N; i++)
    {
        timer.start();
        std::this_thread::sleep_for(sleepTime);
        timer.stop();

        std::cout << "timer data (elapsed: " << timer.getTimeSec() << ")" << std::endl;
        std::cout << "  fps: " << timer.getFPS() << std::endl;
        std::cout << "  average time: " << timer.getAvgTimeSec() << std::endl;
    }

    return 1;
}
