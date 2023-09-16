#pragma once 

#include <iostream>

#define XSTR(x) #x
#define STR(x) XSTR(x)

template <typename T>
inline T sign(const T& x)
{
    return x>=0? T(1) : T(-1);
}

template <typename T>
inline T pow2(const T& x)
{
    return x*x;
}
   

#define MSG_ASSERT(Expr, Msg) __M_Assert(#Expr, Expr, __FILE__, __LINE__, Msg)
void __M_Assert(const char* expr_str, bool expr, const char* file, int line, const char* msg)
{
    if (!expr)
    {
        std::cerr << "Assert failed:\t" << msg << "\n"
            << "Expected:\t" << expr_str << "\n"
            << "Source:\t\t" << file << ", line " << line << "\n";
        abort();
    }
}