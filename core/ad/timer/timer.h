//
// Created by xiang on 2021/11/17.
//

#ifndef FUSION_TIMER_H
#define FUSION_TIMER_H

#include <chrono>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace sad::common {

/// Time profiling utility
/// NOTE Timer seems to have issues when wrapped in gtest
class Timer {
   public:
    struct TimerRecord {
        TimerRecord() = default;
        TimerRecord(const std::string& name, double time_usage) {
            func_name_ = name;
            time_usage_in_ms_.emplace_back(time_usage);
        }
        std::string func_name_;
        std::vector<double> time_usage_in_ms_;
    };

    /**
     * Evaluate and record function runtime
     * @tparam F
     * @param func
     * @param func_name
     */
    template <class F>
    static void Evaluate(F&& func, const std::string& func_name) {
        auto t1 = std::chrono::steady_clock::now();
        std::forward<F>(func)();
        auto t2 = std::chrono::steady_clock::now();
        auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000;

        if (records_.find(func_name) != records_.end()) {
            records_[func_name].time_usage_in_ms_.emplace_back(time_used);
        } else {
            records_.insert({func_name, TimerRecord(func_name, time_used)});
        }
    }

    /// Print all recorded runtimes
    static void PrintAll();

    /// Dump to file for plotting/analysis
    static void DumpIntoFile(const std::string& file_name);

    /// Get average execution time of a function
    static double GetMeanTime(const std::string& func_name);

    /// Clear records
    static void Clear() { records_.clear(); }

   private:
    static std::map<std::string, TimerRecord> records_;
};

}  // namespace sad::utils

#endif  // FUSION_TIMER_H
