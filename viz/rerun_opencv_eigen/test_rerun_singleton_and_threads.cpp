#include <rerun.hpp>

#include <algorithm>  // std::max
#include <cmath>
#include <string>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "rerun_collection_adapters.hpp"
#include "macros.h"

#include "RerunSingleton.h"

rerun::Collection<rerun::TensorDimension> tensor_shape(const cv::Mat& img) {
    return {img.rows, img.cols, img.channels()};
};

void threadImgTask(const std::string& threadName, const std::chrono::steady_clock::time_point& time0) {
    rerun::RecordingStream& rec = RerunSingleton::instance();

    // Define the resolution of the image
    int width = 640;
    int height = 480;

    // Create a blank image with the specified resolution
    cv::Mat randimg(height, width, CV_8UC3);

    int frame_number = 0;

    while (true)
    {
        std::chrono::steady_clock::time_point time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(time - time0);
        const double t = elapsed.count();
        rec.set_time_seconds("timestamp", t);

        //rec.set_time_sequence("frame_number", frame_number++);

        // Generate random pixel values for each channel
        cv::randu(randimg, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        rec.log(threadName + "/randimg", rerun::Image(tensor_shape(randimg), rerun::TensorBuffer::u8(randimg)));

        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
}

void threadPlotTask(const std::string& threadName, const std::chrono::steady_clock::time_point& time0) {
    rerun::RecordingStream& rec = RerunSingleton::instance();

    const double T = 10;                // [s]
    const double omega = 2 * M_PI / T;  // [rad/s]

    while (true)
    {
        std::chrono::steady_clock::time_point time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(time - time0);
        const double t = elapsed.count();
        rec.set_time_seconds("timestamp", t);

        rec.log(threadName + "/plot1", rerun::Scalar(sin(omega * t)));
        rec.log(threadName + "/plot2", rerun::Scalar(cos(omega * t)));

        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
}

int main() {
    std::chrono::steady_clock::time_point time0 = std::chrono::steady_clock::now();

    std::thread t1(threadImgTask, "thread1", time0);
    std::thread t2(threadImgTask, "thread2", time0);
    std::thread t3(threadPlotTask, "thread3", time0);
    std::thread t4(threadPlotTask, "thread4", time0);

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    return 0;
}
