#pragma once

#include <sstream>
#include <string>

namespace slamplay {

template <class Type>
Type string2Num(const std::string &str) {
    std::istringstream iss(str);
    Type num;
    iss >> std::hex >> num;
    return num;
}

}  // namespace slamplay