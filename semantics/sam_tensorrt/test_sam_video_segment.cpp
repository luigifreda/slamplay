

#include "SamInterface.h"

#include "image/image_utils.h"
#include "io/file_utils.h"
#include "io/messages.h"
#include "macros.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>

#include <macros.h>

const std::string kDataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag

///////////////////////////////////////////////////////////////////////////////
using namespace std;
using namespace cv;
using namespace slamplay;

int main(int argc, char** argv) {
    std::string dataset_path;

    if (argc == 2) {
        dataset_path = argv[2];
    } else {
        dataset_path = kDataDir + "/kitti/kitti06/video_color.mp4";
    }

    std::vector<std::string> imagePathList;
    bool isVideo{false};
    if (isFile(dataset_path)) {
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
    } else if (pathExists(dataset_path)) {
        cv::glob(dataset_path + "/*.jpg", imagePathList);
    }

    SamInterface samIf;
    samIf.loadModels();

    auto processImage = [&](const cv::Mat frame) {
        // cout << "Processing input image: " << frame.size() << ", " << cvTypeToStr(frame.type()) << endl;
        cv::Mat mask;
        auto start = std::chrono::system_clock::now();
        samIf.processEmbedding(frame);
        mask = samIf.processAutoSegment(frame);
        auto end = chrono::system_clock::now();
        cout << "Time per output frame : " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
        return mask;
    };

#define USE_RESIZED_FRAME 1

    cv::Mat outImage;
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

#if !USE_RESIZED_FRAME
            cv::Mat& resizedFrame = frame;
#else
            cv::Mat resizedFrame;
            cv::resize(frame, resizedFrame, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
#endif
            cv::Mat mask = processImage(resizedFrame);
            samIf.showAutoSegmentResult(resizedFrame, mask, outImage);
            cv::imshow("auto segmentation", outImage);

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

#if !USE_RESIZED_FRAME
            cv::Mat& resizedFrame = frame;
#else
            cv::Mat resizedFrame;
            cv::resize(frame, resizedFrame, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
#endif

            cv::Mat mask = processImage(resizedFrame);
            samIf.showAutoSegmentResult(resizedFrame, mask, outImage);
            cv::imshow("auto segmentation", outImage);

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