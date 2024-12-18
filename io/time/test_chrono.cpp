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
#include <iostream>
#include <math.h>
#include <chrono>


int main(int argc, char **argv) 
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();	
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    // get time duration in ns 
    // std::chrono::nanoseconds is the most accurate representation of an elapsed time in chrono, as it has the smallest unit of time (one billionth of a second).
    std::chrono::nanoseconds elapsed_ns = (t2-t1); 

    // convert to microseconds duration
    std::chrono::microseconds elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    // or 
    std::chrono::microseconds elapsed2_us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed_ns);

    // convert to double duration 
    std::chrono::duration<double> elapsed_double = std::chrono::duration_cast< std::chrono::duration<double>>(t2 - t1);
    // or 
    std::chrono::duration<double> elapsed2_double = std::chrono::duration_cast< std::chrono::duration<double>>(elapsed_ns);

    // convert to representation std::chrono::nanoseconds::rep 
    // which is the representation type of the std::chrono::nanoseconds class, is a signed integral type of at least 64 bits, which can store the number of ticks of the duration.
    std::cout << "elapsed ns = " << elapsed_ns.count() << "[ns]" << std::endl;       
    
    // convert to representation std::chrono::microseconds::rep 
    // which is the representation type of the std::chrono::microseconds class, is a signed integral type of at least 64 bits
    std::cout << "elapsed us = " << elapsed_us.count() << "[µs]" << std::endl;
    
    // convert to double 
    std::cout << "elapsed double" << elapsed_double.count() << " [s]" << std::endl;    
    
    return 1; 
}
