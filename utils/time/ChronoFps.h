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