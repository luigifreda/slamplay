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