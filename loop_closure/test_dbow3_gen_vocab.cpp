// *************************************************************************
/* 
 * This file is part of the slamplay project.
 * Copyright (C) 2018-present Luigi Freda <luigifreda at gmail dot com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version, at your option. If this file is a modified/adapted 
 * version of an original file distributed under a different license that 
 * is not compatible with the GNU General Public License, the 
 * BSD 3-Clause License will apply instead.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
// *************************************************************************
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>
#include "DBoW3/DBoW3.h"

#include "datasets/TUM.h"
#include "features/FeatureManager.h"
#include "io/file_utils.h"
#include "macros.h"

using namespace cv;
using namespace std;
using namespace slamplay;

std::string dataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag

/***************************************************
* This example demonstrates how to train a dictionary
  from the images available in the input dataset directory
*************************************************/

int main(int argc, char** argv) {
    string dataset_dir = dataDir + "/loop_closure/";
    if (argc == 2) {
        dataset_dir = argv[1];
    } else {
        cout << "usage: " << argv[0] << " <dataset dir>" << endl;
    }

    std::vector<std::string> filenames;
    getImageFilenames(dataset_dir, filenames);

    cout << "extracting features ... " << endl;
    const string feature_type = "orb";
    cout << "type: " << feature_type << endl;
    Ptr<Feature2D> detector = getFeature2D(feature_type);
    int index = 1;
    vector<Mat> descriptors;
    for (const string& file : filenames)
    {
        Mat image = imread(file);
        vector<KeyPoint> keypoints;
        Mat descriptor;
        detector->detectAndCompute(image, Mat(), keypoints, descriptor);
        descriptors.push_back(descriptor);
        cout << "extracting features from image " << index++ << " " << file << endl;
    }
    cout << "extracted total " << descriptors.size() * 500 << " features." << endl;

    // create vocabulary
    cout << "creating vocabulary, please wait ... " << endl;
    DBoW3::Vocabulary vocab;
    vocab.create(descriptors);
    cout << "vocabulary info: " << vocab << endl;
    vocab.save("vocab.yml.gz");
    cout << "done" << endl;

    return 0;
}