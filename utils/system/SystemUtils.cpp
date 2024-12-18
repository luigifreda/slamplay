#include "system/SystemUtils.h"

#include <fstream>

#if __APPLE__
#include <mach/mach.h>
#include <mach/task.h>
#elif __linux__
#include <unistd.h>

#include <sys/resource.h>
#endif

namespace slamplay {

bool get_memory_info(MemoryInfo& info) {
#if __APPLE__
    mach_task_basic_info_data_t t_info;
    mach_msg_type_number_t t_info_count = MACH_TASK_BASIC_INFO_COUNT;

    if (KERN_SUCCESS != task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                                  (task_info_t)&t_info, &t_info_count)) {
        return false;
    }
    info.resident_memory = t_info.resident_size;
    info.resident_memory_peak = t_info.resident_size_max;

    /*
    struct rusage resource_usage;
    getrusage(RUSAGE_SELF, &resource_usage);
    info.resident_memory_peak = resource_usage.ru_maxrss;
    */

    return true;
#elif __linux__

    // get current memory first
    std::size_t program_size = 0;
    std::size_t resident_size = 0;

    std::ifstream fs("/proc/self/statm");
    if (fs.fail()) {
        return false;
    }
    fs >> program_size;
    fs >> resident_size;

    info.resident_memory = resident_size * sysconf(_SC_PAGESIZE);

    // get peak memory after that
    struct rusage resource_usage;
    getrusage(RUSAGE_SELF, &resource_usage);
    info.resident_memory_peak = resource_usage.ru_maxrss * 1024;

    return true;
#else
    return false;
#endif
}

}  // namespace slamplay
