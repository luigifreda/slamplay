#pragma once

#include <rerun.hpp>
#include <string>

class RerunSingleton {
   public:
    static rerun::RecordingStream& instance(std::string instanceName = "my_rerun") {
        static rerun::RecordingStream rec = rerun::RecordingStream(instanceName);
        static bool first = true;
        if (first) {
            first = false;
            rec.spawn().exit_on_failure();
        }
        return rec;
    }
};