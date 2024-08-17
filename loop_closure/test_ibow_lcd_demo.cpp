/**
 * This file is part of ibow-lcd.
 *
 * Copyright (C) 2017 Emilio Garcia-Fidalgo <emilio.garcia@uib.es> (University of the Balearic Islands)
 *
 * ibow-lcd is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ibow-lcd is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ibow-lcd. If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>

#include <boost/filesystem.hpp>
#include <opencv2/features2d.hpp>

#include "ibow-lcd/lcdetector.h"

#include "io/file_utils.h"
#include "macros.h"

using namespace slamplay;

std::string dataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag

int main(int argc, char** argv) {
    // std::string dataset_dir = dataDir + "/loop_closure/";
    std::string dataset_dir = dataDir + "/new_college/Images";
    if (argc == 2) {
        dataset_dir = argv[1];
    } else {
        std::cout << "usage: " << argv[0] << " <dataset dir>" << std::endl;
    }

    // Creating feature detector and descriptor
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create(1500);  // Default params

    // Loading image filenames
    std::vector<std::string> filenames;
    getImageFilenames(dataset_dir, filenames);
    unsigned nimages = filenames.size();

    // Creating the loop closure detector object
    ibow_lcd::LCDetectorParams params;  // Assign desired parameters
    ibow_lcd::LCDetector lcdet(params);

    // Processing the sequence of images
    for (unsigned i = 0; i < nimages; i++) {
        // Processing image i
        std::cout << "--- Processing image " << i << std::endl;

        // Loading and describing the image
        cv::Mat img = cv::imread(filenames[i]);
        std::vector<cv::KeyPoint> kps;
        detector->detect(img, kps);
        cv::Mat dscs;
        detector->compute(img, kps, dscs);

        ibow_lcd::LCDetectorResult result;
        lcdet.process(i, kps, dscs, &result);

        switch (result.status)
        {
            case ibow_lcd::LC_DETECTED:
                std::cout << "--- Loop detected!!!: " << result.train_id << " with " << result.inliers << " inliers" << std::endl;
                break;
            case ibow_lcd::LC_NOT_DETECTED:
                std::cout << "No loop found" << std::endl;
                break;
            case ibow_lcd::LC_NOT_ENOUGH_IMAGES:
                std::cout << "Not enough images to found a loop" << std::endl;
                break;
            case ibow_lcd::LC_NOT_ENOUGH_ISLANDS:
                std::cout << "Not enough islands to found a loop" << std::endl;
                break;
            case ibow_lcd::LC_NOT_ENOUGH_INLIERS:
                std::cout << "Not enough inliers" << std::endl;
                break;
            case ibow_lcd::LC_TRANSITION:
                std::cout << "Transitional loop closure" << std::endl;
                break;
            default:
                std::cout << "No status information" << std::endl;
                break;
        }
    }

    return 0;
}
