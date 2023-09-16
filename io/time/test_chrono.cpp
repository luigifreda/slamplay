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
