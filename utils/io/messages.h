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

#include <iostream>
#include <sstream>

namespace IoColor {
enum Code {
    FG_RED = 31,
    FG_GREEN = 32,
    FG_YELLOW = 33,
    FG_BLUE = 34,
    FG_DEFAULT = 39,
    BG_RED = 41,
    BG_GREEN = 42,
    BG_YELLOW = 43,
    BG_BLUE = 44,
    BG_DEFAULT = 49
};

class Modifier {
    Code code;

   public:
    Modifier(Code pCode) : code(pCode) {
    }
    friend std::ostream& operator<<(std::ostream& os, const Modifier& mod) {
        return os << "\033[" << mod.code << "m";
    }
};

inline const Modifier Default() {
    return Modifier(FG_DEFAULT);
}
inline const Modifier Red() {
    return Modifier(FG_RED);
}
inline const Modifier Green() {
    return Modifier(FG_GREEN);
}
inline const Modifier Blue() {
    return Modifier(FG_BLUE);
}
inline const Modifier Yellow() {
    return Modifier(FG_YELLOW);
}
}  // namespace IoColor

#define MSG_ASSERT(condition, message)                                                 \
    {                                                                                  \
        if (!(condition))                                                              \
        {                                                                              \
            const IoColor::Modifier red = IoColor::Red();                              \
            const IoColor::Modifier def = IoColor::Default();                          \
            std::cerr << red << "Assertion failed at " << __FILE__ << ":" << __LINE__; \
            std::cerr << " inside " << __FUNCTION__ << def << std::endl;               \
            std::cerr << "Condition: " << #condition << std::endl;                     \
            std::cerr << "Message: " << message << std::endl;                          \
            std::abort();                                                              \
        }                                                                              \
    }

#define MSG_LOG(message, prefix, modifier)                              \
    {                                                                   \
        const IoColor::Modifier def = IoColor::Default();               \
        std::cerr << modifier << prefix << message << def << std::endl; \
    }

#define MSG_ERROR(message)                          \
    {                                               \
        MSG_LOG(message, "ERROR: ", IoColor::Red()) \
        std::abort();                               \
    }

#define MSG_ERROR_STREAM(args) \
    do                         \
    {                          \
        std::stringstream ss;  \
        ss << args;            \
        MSG_ERROR(ss.str())    \
    } while (0)

#define MSG_INFO(message)                           \
    {                                               \
        MSG_LOG(message, "INFO: ", IoColor::Blue()) \
    }

#define MSG_INFO_STREAM(args) \
    do                        \
    {                         \
        std::stringstream ss; \
        ss << args;           \
        MSG_INFO(ss.str())    \
    } while (0)

#define MSG_WARN(message)                                \
    {                                                    \
        MSG_LOG(message, "WARNING: ", IoColor::Yellow()) \
    }

#define MSG_WARN_STREAM(args) \
    do                        \
    {                         \
        std::stringstream ss; \
        ss << args;           \
        MSG_WARN(ss.str())    \
    } while (0)
