#include <iostream>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rerun.hpp>

#include <thread>

#include "macros.h"
#include "viz/rerun_collection_adapters.hpp"

std::string dataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag

rerun::Collection<rerun::TensorDimension> tensor_shape(const cv::Mat& img) {
    return {img.rows, img.cols, img.channels()};
};

int main() {
    std::cout << "Rerun SDK Version: " << rerun::version_string() << std::endl;

    const auto rec = rerun::RecordingStream("rerun_example_images");
    rec.spawn().exit_on_failure();

    // Read image
    const auto image_path = dataDir + "/rerun-logo.png";
    cv::Mat img = imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    // Rerun expects RGB format
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Log image to rerun using the tensor buffer adapter defined in `collection_adapters.hpp`.
    rec.log("images1/image0", rerun::Image(tensor_shape(img), rerun::TensorBuffer::u8(img)));

    // Or by passing a pointer to the image data.
    // The pointer cast here is redundant since `data` is already uint8_t in this case, but if you have e.g. a float image it may be necessary to cast to float*.
    rec.log("images2/image1", rerun::Image(tensor_shape(img), reinterpret_cast<const uint8_t*>(img.data)));

    // Define the resolution of the image
    int width = 640;
    int height = 480;

    // Create a blank image with the specified resolution
    cv::Mat randomImage(height, width, CV_8UC3);

    while (true)
    {
        // Generate random pixel values for each channel
        cv::randu(randomImage, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        rec.log("images_rand/image_rand", rerun::Image(tensor_shape(randomImage), rerun::TensorBuffer::u8(randomImage)));

        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    return 0;
}
