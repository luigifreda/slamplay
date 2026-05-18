//
// Created by xiang on 2021/7/21.
//

#ifndef SLAM_IN_AUTO_DRIVING_SYS_UTILS_H
#define SLAM_IN_AUTO_DRIVING_SYS_UTILS_H

#include <glog/logging.h>
#include <chrono>

namespace sad {

// Some system-related utilities, such as code timing

/**
 * Measure code execution time
 * @tparam FuncT
 * @param func  Function to call
 * @param func_name Function name
 * @param times Number of calls
 */
template <typename FuncT>
void evaluate_and_call(FuncT&& func, const std::string& func_name = "", int times = 10) {
    double total_time = 0;
    for (int i = 0; i < times; ++i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        func();
        auto t2 = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000;
    }

    LOG(INFO) << "Method " << func_name << " average call time/count: " << total_time / times << "/" << times << " ms.";
}

}  // namespace sad

#endif  // SLAM_IN_AUTO_DRIVING_SYS_UTILS_H
