
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "PointCloudViz.h"
#include "pointcloud_from_image_depth.h"

#include <cv/matches_utils.h>
#include <image/image_depth.h>
#include <image/image_utils.h>
#include <macros.h>
#include <messages.h>
#include "depth_anything/DepthAnything.h"
#include "depth_anything/DepthAnythingSettings.h"

#include <NvInfer.h>
#include <sys/stat.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>

const std::string kDataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag

using namespace std;
using namespace slamplay;

// Define the format used by the point cloud: XYZRGB is used here
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

// Helper function to replace all occurrences of a character in a string
void replaceChar(std::string& str, char find, char replace) {
    size_t pos = 0;
    while ((pos = str.find(find, pos)) != std::string::npos) {
        str[pos] = replace;
        pos++;
    }
}

bool IsPathExist(const std::string& path) {
    return (access(path.c_str(), F_OK) == 0);
}
bool IsFile(const std::string& path) {
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }

    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

bool createFolder(const std::string& folderPath) {
    if (mkdir(folderPath.c_str(), 0777) != 0) {
        if (errno == EEXIST) {
            std::cout << "Folder already exists!" << std::endl;
            return true;  // Folder already exists
        } else {
            std::cerr << "Failed to create folder! Error code: " << errno << std::endl;
            return false;  // Failed to create folder
        }
    }

    std::cout << "Folder created successfully!" << std::endl;
    return true;  // Folder created successfully
}

/**
 * @brief Setting up Tensorrt logger
 */
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

#define ENABLE_PANGOLIN_CLOUD_VIZ 1

int main(int argc, char** argv) {
    std::string engine_file_path;
    std::string dataset_path;
    float depthScale = 1.0;

#if 1
    const std::string configFile = kDataDir + "/depth_anything/config_hypersim.yaml";
#else
    const std::string configFile = kDataDir + "/depth_anything/config_kitti.yaml";
#endif

    DepthAnythingSettings settings(configFile);
    cout << "Settings: " << settings << std::endl;

    if (argc == 3) {
        engine_file_path = argv[1];
        dataset_path = argv[2];
    } else {
        engine_file_path = settings.strModelPath();
        dataset_path = settings.strDatasetPath();
        depthScale = settings.depthScale();
    }

#if ENABLE_PANGOLIN_CLOUD_VIZ
    PointCloudViz<PointCloudT> viz;
    viz.start();
#endif

    std::vector<std::string> imagePathList;
    bool isVideo{false};
    if (IsFile(dataset_path)) {
        std::string suffix = dataset_path.substr(dataset_path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png")
        {
            imagePathList.push_back(dataset_path);
        } else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov" || suffix == "mkv")
        {
            isVideo = true;
        } else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    } else if (IsPathExist(dataset_path)) {
        cv::glob(dataset_path + "/*.jpg", imagePathList);
    }
    // Assume it's a folder, add logic to handle folders
    // init model
    cout << "Loading model from " << engine_file_path << "..." << endl;
    DepthAnything depth_model(engine_file_path, logger);
    cout << "The model has been successfully loaded!" << endl;

    PointCloudT pointcloud;
    const std::vector<float> cameraParams = settings.cameraParams();
    const double fx = cameraParams[0];
    const double fy = cameraParams[1];
    const double cx = cameraParams[2];
    const double cy = cameraParams[3];
    const slamplay::Intrinsics intrinsics{fx, fy, cx, cy};
    const Eigen::Isometry3d Twc = Eigen::Isometry3d::Identity();
    const int border = 0;     // edge width
    const cv::Mat emptyMask;  // empty mask

    auto processImage = [&](const cv::Mat frame, cv::Mat& result_d) {
        // cout << "Predicting for input image: " << frame.size() << ", " << cvTypeToStr(frame.type()) << endl;

        auto start = std::chrono::system_clock::now();
        result_d = depth_model.predict(frame);
        auto end = chrono::system_clock::now();
        cout << "Time per output frame (" << result_d.size << ", " << cvTypeToStr(result_d.type()) << "): " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

        // Resize predicted depth image to the same size as the input image with nearest neighbor interpolation
        cv::resize(result_d, result_d, cv::Size(frame.cols, frame.rows), 0, 0, cv::INTER_NEAREST);
        // cout << "Scaled output image: " << result_d.size() << ", " << cvTypeToStr(result_d.type()) << endl;

        cv::Mat result_d_gray = convertDepthImageToGray(result_d);
        cv::Mat result_d_rgb;
        cv::applyColorMap(result_d_gray, result_d_rgb, cv::COLORMAP_INFERNO);

        cv::Mat result;
        cv::hconcat(frame, result_d_rgb, result);

#if ENABLE_PANGOLIN_CLOUD_VIZ
        // here, depth represents the distance |OP| from camera center to 3D point P
        getPointCloudFromImageAndDepth<PointT, float>(frame, result_d, emptyMask, intrinsics, border, Twc, pointcloud);
        viz.update(pointcloud);
#endif

        return result;
    };

    if (isVideo) {
        // path to video
        string VideoPath = dataset_path;
        // open cap
        cv::VideoCapture cap(VideoPath);

        int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        // Create a VideoWriter object to save the processed video
        // cv::VideoWriter output_video("output_video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(width, height));
        while (1)
        {
            cv::Mat frame;
            cap >> frame;

            if (frame.empty())
                break;

            cv::Mat result_d;
            cv::Mat result = processImage(frame, result_d);
            cv::imshow("depth_result", result);
            cv::waitKey(1);
        }

        // Release resources
        cv::destroyAllWindows();
        cap.release();
        // output_video.release();
    } else {
        // path to folder saves images
        // string imageFolderPath_out = "results/";
        // createFolder(imageFolderPath_out);
        for (const auto& imagePath : imagePathList)
        {
            // open image
            cv::Mat frame = cv::imread(imagePath);
            if (frame.empty())
            {
                cerr << "Error reading image: " << imagePath << endl;
                continue;
            }

            cv::Mat result_d;
            cv::Mat result = processImage(frame, result_d);

            cv::waitKey(1);

            std::istringstream iss(imagePath);
            std::string token;
            while (std::getline(iss, token, '/')) {
            }
            token = token.substr(token.find_last_of("/\\") + 1);

            // std::cout << "Path : " << imageFolderPath_out + token << std::endl;
            // cv::imwrite(imageFolderPath_out + token, result_d);
        }
    }

    cout << "finished" << endl;
    return 0;
}