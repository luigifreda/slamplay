#pragma once

#include <cstdint>
#include <string>

namespace slamplay {

struct MemoryInfo {
    uint64_t resident_memory = 0;       //!< in bytes
    uint64_t resident_memory_peak = 0;  //!< in bytes
};

bool get_memory_info(MemoryInfo& info);

}  // namespace slamplay
