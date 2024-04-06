#pragma once

#include <iostream>

#define XSTR(x) #x
#define STR(x) XSTR(x)

template <typename T>
inline T sign(const T& x) {
    return x >= 0 ? T(1) : T(-1);
}

template <typename T>
inline T pow2(const T& x) {
    return x * x;
}