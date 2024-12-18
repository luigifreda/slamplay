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
#pragma once

#include <chrono>
#include <thread>

namespace slamplay {

class ChronoFps {
   public:
    // Constructor that takes the sleep duration as a parameter
    explicit ChronoFps(double d) : sleep_duration(d * std::chrono::seconds{1}), last_call(std::chrono::steady_clock::now()), first(false) {}
    explicit ChronoFps(std::chrono::duration<double> d) : sleep_duration(d), last_call(std::chrono::steady_clock::now()), first(false) {}

    // Method that sleeps for the remaining time until the next call
    void sleep() {
        if (!first) {
            // Get the current time
            auto now = std::chrono::steady_clock::now();
            // Calculate the elapsed time since the last call
            auto elapsed = now - last_call;
            // If the elapsed time is less than the sleep duration, sleep for the difference
            if (elapsed < sleep_duration) {
                std::this_thread::sleep_for(sleep_duration - elapsed);
            }
        } else {
            first = false;
        }

        // Update the last call time
        last_call = std::chrono::steady_clock::now();
    }

   private:
    // The duration to sleep between calls
    std::chrono::duration<double> sleep_duration;
    // The last time the sleep method was called
    std::chrono::time_point<std::chrono::steady_clock> last_call;

    bool first{true};
};

}  // namespace slamplay    