#pragma once

// Define a macro to create a version number in a more readable format
#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#define GCC_VERSION_STRING #__GNUC__ "." #__GNUC_MINOR__ "." #__GNUC_PATCHLEVEL__

#if GCC_VERSION >= 130200
#include <fmt/format.h>
namespace fmt {
namespace v6 = v9;
}  // namespace fmt
#elif GCC_VERSION >= 110000
#include <fmt/format.h>
namespace fmt {
namespace v6 = v8;
}
#endif