//
// Created by haoyuefan on 2021/9/22.
//

#include <chrono>
#include <memory>

#include "features_dl/superpointglue_tensorrt/SuperGlue.h"
#include "features_dl/superpointglue_tensorrt/SuperPoint.h"

#include "io/file_utils.h"
#include "viz/viz_matches.h"

#include "macros.h"

std::string dataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag

static void help() {
    std::cout << "./test_superpointglue_sequence <config_path> <model_dir> <image_folder_absolutely_path> <output_folder_path>" << std::endl;
}

int main(int argc, char** argv) {
    std::string config_path = dataDir + "/superpointglue/config/config.yaml";
    std::string model_dir = dataDir + "/superpointglue/weights/";
    std::string image_path = dataDir + "/freiburg_sequence";
    std::string output_path = dataDir + "/freiburg_sequence/matches";

    if (argc == 5)
    {
        config_path = argv[1];
        model_dir = argv[2];
        image_path = argv[3];
        output_path = argv[4];
    } else if (argc < 5)
    {
        help();
    }

    std::vector<std::string> image_names;
    slamplay::getFilenames(image_path, image_names);
    Configs configs(config_path, model_dir);
    int width = configs.superglue_config.image_width;
    int height = configs.superglue_config.image_height;

    std::cout << "Building inference engine for superpoint......" << std::endl;
    auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
    if (!superpoint->build()) {
        std::cerr << "Error in SuperPoint building engine. Please check your onnx model path." << std::endl;
        return 0;
    }
    std::cout << "Building inference engine for superglue......" << std::endl;
    auto superglue = std::make_shared<SuperGlue>(configs.superglue_config);
    if (!superglue->build()) {
        std::cerr << "Error in SuperGlue building engine. Please check your onnx model path." << std::endl;
        return 0;
    }
    std::cout << "SuperPoint and SuperGlue inference engine build success." << std::endl;

    Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points0;
    cv::Mat image0 = cv::imread(image_names[0], cv::IMREAD_GRAYSCALE);
    if (image0.empty()) {
        std::cerr << "First image in the image folder is empty." << std::endl;
        return 0;
    }
    cv::resize(image0, image0, cv::Size(width, height));
    std::cout << "First image size: " << image0.cols << "x" << image0.rows << std::endl;
    if (!superpoint->infer(image0, feature_points0)) {
        std::cerr << "Failed when extracting features from first image." << std::endl;
        return 0;
    }
    std::vector<cv::DMatch> init_matches;
    superglue->matching_points(feature_points0, feature_points0, init_matches);
    std::string mkdir_cmd = "mkdir -p " + output_path;
    system(mkdir_cmd.c_str());

    cv::namedWindow("match_image", cv::WINDOW_NORMAL);

    for (int index = 1; index < image_names.size(); ++index) {
        Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points1;
        std::vector<cv::DMatch> superglue_matches;
        cv::Mat image1 = cv::imread(image_names[index], cv::IMREAD_GRAYSCALE);
        if (image1.empty()) continue;
        cv::resize(image1, image1, cv::Size(width, height));

        std::cout << "Second image size: " << image1.cols << "x" << image1.rows << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        if (!superpoint->infer(image1, feature_points1)) {
            std::cerr << "Failed when extracting features from second image." << std::endl;
            return 0;
        }
        superglue->matching_points(feature_points0, feature_points1, superglue_matches);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
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

        slamplay::visualizeMatches(image0, keypoints0, image1, keypoints1, superglue_matches, match_image, duration.count());
        cv::imshow("match_image", match_image);

        std::cout << "press a key to proceed..." << std::endl;
        cv::waitKey(-1);

        cv::imwrite(output_path + "/" + std::to_string(index) + ".png", match_image);
    }

    return 0;
}
