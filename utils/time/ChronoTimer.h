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

#include <math.h>
#include <chrono>
#include <iostream>
#include <limits>

namespace slamplay {

/*
The class computes passing time by counting the number of ticks per second.
That is, the following code computes the execution time in seconds.
It is also possible to compute the average time over multiple runs.
*/
class ChronoTimer {
   public:
    //! the default constructor
    ChronoTimer() {
        reset();
    }

    //! starts counting ticks.
    void start() {
        startTime = std::chrono::steady_clock::now();
    }

    //! stops counting ticks.
    void stop() {
        const std::chrono::steady_clock::time_point time = std::chrono::steady_clock::now();
        if (startTime.time_since_epoch().count() == 0)
            return;
        ++counter;
        sumTime += (time - startTime);
        startTime = std::chrono::steady_clock::time_point{};
    }

    //! returns counted ticks.
    //  The type rep, which is the representation type of the std::chrono::nanoseconds class, is a signed integral type of at least 64 bits, which can store the number of ticks of the duration.
    std::chrono::nanoseconds::rep getTimeTicks() const {
        return sumTime.count();
    }

    //! returns passed time in microseconds.
    double getTimeMicro() const {
        return getTimeMilli() * 1e3;
    }

    //! returns passed time in milliseconds.
    double getTimeMilli() const {
        return getTimeSec() * 1e3;
    }

    //! returns passed time in seconds.
    double getTimeSec() const {
        return std::chrono::duration_cast<std::chrono::duration<double>>(sumTime).count();
    }

    //! returns internal counter value.
    uint64_t getCounter() const {
        return counter;
    }

    //! returns average FPS (frames per second) value.
    double getFPS() const {
        const double sec = getTimeSec();
        if (sec < std::numeric_limits<double>::epsilon())
            return 0.;
        return counter / sec;
    }

    //! returns average time in seconds
    double getAvgTimeSec() const {
        if (counter <= 0)
            return 0.;
        return getTimeSec() / counter;
    }

    //! returns average time in milliseconds
    double getAvgTimeMilli() const {
        return getAvgTimeSec() * 1e3;
    }

    //! resets internal values.
    void reset() {
        startTime = std::chrono::steady_clock::time_point{};
        sumTime = std::chrono::nanoseconds{0};
        counter = 0;
    }

   private:
    uint64_t counter{0};
    std::chrono::nanoseconds sumTime{0};                // std::chrono::nanoseconds is the most accurate representation of an elapsed time in chrono, as it has the smallest unit of time (one billionth of a second).
    std::chrono::steady_clock::time_point startTime{};  // default value of zero
};

}  // namespace slamplay