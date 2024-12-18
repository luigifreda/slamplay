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

#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>
#include <opencv2/opencv.hpp>

namespace slamplay {

struct TicToc {
    TicToc() {};
    void clearBuff() { timeBuff.clear(); }
    void Tic() { t1 = std::chrono::steady_clock::now(); }
    float Toc() {
        t2 = std::chrono::steady_clock::now();
        float time = std::chrono::duration<float, std::milli>(t2 - t1).count();
        timeBuff.emplace_back(time);
        return time;
    }
    float aveCost(void) {
        if (timeBuff.empty()) return 0;
        return std::accumulate(timeBuff.begin(), timeBuff.end(), 0.f) / (float)timeBuff.size();
    }
    float devCost(void) {
        if (timeBuff.size() <= 1) return 0;
        float average = aveCost();

        float accum = 0;
        int total = 0;
        for (double value : timeBuff)
        {
            if (value == 0)
                continue;
            accum += pow(value - average, 2);
            total++;
        }
        return sqrt(accum / total);
    }
    bool empty() { return timeBuff.empty(); }

    std::vector<float> timeBuff;
    std::chrono::steady_clock::time_point t1;
    std::chrono::steady_clock::time_point t2;
};

}  // namespace slamplay