#include <iostream>
#include <math.h>
#include <thread>
#include <unistd.h>

#include "ChronoTimer.h"

using namespace std::chrono_literals;

int main(int argc, char **argv) 
{
    ChronoTimer timer; 

    int N = 1000; 
    auto sleepTime = 10ms; // can write this thanks to std::chrono_literals;

    for(int i=0; i<N; i++)
    {
        timer.start(); 
        std::this_thread::sleep_for(sleepTime);
        timer.stop(); 

        std::cout << "timer data (elapsed: " << timer.getTimeSec() <<")" << std::endl; 
        std::cout << "  fps: " << timer.getFPS() << std::endl;         
        std::cout << "  average time: " << timer.getAvgTimeSec() << std::endl;
    }

    return 1; 
}
