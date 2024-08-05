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