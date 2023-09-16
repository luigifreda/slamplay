#pragma once 

#include <string> 
#include <algorithm>
#include <sstream>


std::string str_tolower(const std::string& s)
{
    std::string out = s.c_str();  
    std::transform(out.begin(), out.end(), out.begin(), 
                   [](unsigned char c){ return std::tolower(c); } 
                  );
    return out;
}


template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return std::move(out).str();
}