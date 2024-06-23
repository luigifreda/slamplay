//
// Created by haoyuefan on 2021/9/22.
//

#include <chrono>
#include <memory>
#include "super_glue.h"
#include "super_point.h"
#include "utils.h"

#include "macros.h"

std::string dataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag

static void help() {
    std::cout << "\n./test_superpointglue_image <config_path> <model_dir> <first_image_absolutely_path> <second_image_absolutely_path>" << std::endl;
}

int main(int argc, char** argv) {
    std::string image0_path = dataDir + "/superpointglue/image0.png";
    std::string image1_path = dataDir + "/superpointglue/image1.png";

    std::string config_path = dataDir + "/superpointglue/config/config.yaml";
    std::string model_dir = dataDir + "/superpointglue/weights/";

    if (argc == 5)
    {
        config_path = argv[1];
        model_dir = argv[2];
        image0_path = argv[3];
        image1_path = argv[4];
    } else if (argc < 5)
    {
        help();
    }

    cv::Mat image0 = cv::imread(image0_path, cv::IMREAD_GRAYSCALE);
    cv::Mat image1 = cv::imread(image1_path, cv::IMREAD_GRAYSCALE);

    if (image0.empty() || image1.empty()) {
        std::cerr << "Input image is empty. Please check the image path." << std::endl;
        return 0;
    }

    Configs configs(config_path, model_dir);
    int width = configs.superglue_config.image_width;
    int height = configs.superglue_config.image_height;

    cv::resize(image0, image0, cv::Size(width, height));
    cv::resize(image1, image1, cv::Size(width, height));
    std::cout << "First image size: " << image0.cols << "x" << image0.rows << std::endl;
    std::cout << "Second image size: " << image1.cols << "x" << image1.rows << std::endl;

    std::cout << "Building inference engine......" << std::endl;
    auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
    if (!superpoint->build()) {
        std::cerr << "Error in SuperPoint building engine. Please check your onnx model path." << std::endl;
        return 0;
    }
    auto superglue = std::make_shared<SuperGlue>(configs.superglue_config);
    if (!superglue->build()) {
        std::cerr << "Error in SuperGlue building engine. Please check your onnx model path." << std::endl;
        return 0;
    }
    std::cout << "SuperPoint and SuperGlue inference engine build success." << std::endl;

    Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points0, feature_points1;
    std::vector<cv::DMatch> superglue_matches;

    double image0_tcount = 0;
    double image1_tcount = 0;
    double match_tcount = 0;

    constexpr int num_times = 10;
    std::cout << "SuperPoint and SuperGlue test in " << num_times << " times." << std::endl;

    cv::namedWindow("match_image", cv::WINDOW_NORMAL);

    for (int i = 0; i <= num_times; ++i) {
        std::cout << "---------------------------------------------------------" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        if (!superpoint->infer(image0, feature_points0)) {
            std::cerr << "Failed when extracting features from first image." << std::endl;
            return 0;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (i > 0) {
            std::cout << "First image feature points number: " << feature_points0.cols() << std::endl;
            image0_tcount += duration.count();
            std::cout << "First image infer cost " << image0_tcount / i << " MS" << std::endl;
        }
        start = std::chrono::high_resolution_clock::now();
        if (!superpoint->infer(image1, feature_points1)) {
            std::cerr << "Failed when extracting features from second image." << std::endl;
            return 0;
        }
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (i > 0) {
            std::cout << "Second image feature points number: " << feature_points1.cols() << std::endl;
            image1_tcount += duration.count();
            std::cout << "Second image infer cost " << image1_tcount / i << " MS" << std::endl;
        }

        start = std::chrono::high_resolution_clock::now();
        superglue->matching_points(feature_points0, feature_points1, superglue_matches);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (i > 0) {
            match_tcount += duration.count();
            std::cout << "Match image cost " << match_tcount / i << " ms" << std::endl;
        }
    }

    cv::Mat match_image;
    std::vector<cv::KeyPoint> keypoints0, keypoints1;
    for (size_t i = 0; i < feature_points0.cols(); ++i) {
        double score = feature_points0(0, i);
        double x = feature_points0(1, i);
        double y = feature_points0(2, i);
        keypoints0.emplace_back(x, y, 8, -1, score);
    }
    for (size_t i = 0; i < feature_points1.cols(); ++i) {
        double score = feature_points1(0, i);
        double x = feature_points1(1, i);
        double y = feature_points1(2, i);
        keypoints1.emplace_back(x, y, 8, -1, score);
    }

    cv::drawMatches(image0, keypoints0, image1, keypoints1, superglue_matches, match_image);
    cv::imwrite("match_image.png", match_image);
    //  visualize
    cv::imshow("match_image", match_image);
    std::cout << "press a key to proceed..." << std::endl;
    cv::waitKey(-1);

    return 0;
}
