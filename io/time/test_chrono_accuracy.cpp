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
#include <chrono>
#include <iomanip>
#include <iostream>

template <typename T>
void printRatio()
{ 
    std::cout << "  precision: " << T::num << "/" << T::den << " second " << std::endl;
    typedef typename std::ratio_multiply<T,std::kilo>::type MillSec;
    typedef typename std::ratio_multiply<T,std::mega>::type MicroSec;
    std::cout << std::fixed;
    std::cout << "             " << static_cast<double>(MillSec::num)/MillSec::den << " milliseconds " << std::endl;
    std::cout << "             " << static_cast<double>(MicroSec::num)/MicroSec::den << " microseconds " << std::endl;
}

int main(){
    
    std::cout << std::boolalpha << std::endl;
    
    std::cout << "std::chrono::system_clock: " << std::endl;
    std::cout << "  is steady: " << std::chrono::system_clock::is_steady << std::endl;
    printRatio<std::chrono::system_clock::period>();
    
    std::cout << std::endl;
    
    std::cout << "std::chrono::steady_clock: " << std::endl;
    std::cout << "  is steady: " << std::chrono::steady_clock::is_steady << std::endl;
    printRatio<std::chrono::steady_clock::period>();
    
    std::cout << std::endl;
    
    std::cout << "std::chrono::high_resolution_clock: " << std::endl;
    std::cout << "  is steady: " << std::chrono::high_resolution_clock::is_steady << std::endl;
    printRatio<std::chrono::high_resolution_clock::period>();
    
    
    std::cout << std::endl;
    
}