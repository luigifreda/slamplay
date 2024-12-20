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
/**
 * Date:  2016
 * Author: Rafael Muñoz Salinas
 * Description: demo application of DBoW3
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>

// DBoW3
#include "DBoW3/DBoW3.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#ifdef HAVE_OPENCV_CONTRIB
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#endif
#include "DBoW3/DescManip.h"

#include "io/CmdLineParser.h"
#include "io/file_utils.h"

using namespace DBoW3;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void wait() {
    cout << endl
         << "Press enter to continue" << endl;
    getchar();
}

vector<string> readImagePaths(int argc, char **argv, int start) {
    vector<string> paths;
    for (int i = start; i < argc; i++) paths.push_back(argv[i]);
    return paths;
}

vector<cv::Mat> extractFeatures(std::vector<string> path_to_images, string descriptor = "") /*throw (std::exception)*/
{
    // select detector
    cv::Ptr<cv::Feature2D> fdetector;
    if (descriptor == "orb")
        fdetector = cv::ORB::create();
    else if (descriptor == "brisk")
        fdetector = cv::BRISK::create();
    else if (descriptor == "akaze")
        fdetector = cv::AKAZE::create();
#ifdef HAVE_OPENCV_CONTRIB
    else if (descriptor == "surf")
        fdetector = cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
#endif
    else
        throw std::runtime_error("Invalid descriptor");
    assert(!descriptor.empty());

    vector<cv::Mat> features;

    cout << "Extracting features..." << endl;
    for (size_t i = 0; i < path_to_images.size(); ++i)
    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        cout << "reading image: " << path_to_images[i] << endl;
        cv::Mat image = cv::imread(path_to_images[i], 0);
        if (image.empty()) throw std::runtime_error("Could not open image" + path_to_images[i]);
        cout << "extracting features" << endl;
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        features.push_back(descriptors);
        cout << "done detecting features" << endl;
    }
    return features;
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<cv::Mat> &features) {
    // branching factor and depth levels
    const int k = 9;
    const int L = 3;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;

    DBoW3::Vocabulary voc(k, L, weight, score);

    cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;

    cout << "Vocabulary information: " << endl
         << voc << endl
         << endl;

    // lets do something with this vocabulary
    cout << "Matching images against themselves (0 low, 1 high): " << endl;
    BowVector v1, v2;
    for (size_t i = 0; i < features.size(); i++)
    {
        voc.transform(features[i], v1);
        for (size_t j = 0; j < features.size(); j++)
        {
            voc.transform(features[j], v2);

            double score = voc.score(v1, v2);
            cout << "Image " << i << " vs Image " << j << ": " << score << endl;
        }
    }

    // save the vocabulary to disk
    cout << endl
         << "Saving vocabulary..." << endl;
    voc.save("small_voc.yml.gz");
    cout << "Done" << endl;
}

////// ----------------------------------------------------------------------------

void testDatabase(const vector<cv::Mat> &features) {
    cout << "Creating a small database..." << endl;

    // load the vocabulary from disk
    Vocabulary voc("small_voc.yml.gz");

    Database db(voc, false, 0);  // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now

    // add images to the database
    for (size_t i = 0; i < features.size(); i++)
        db.add(features[i]);

    cout << "... done!" << endl;

    cout << "Database information: " << endl
         << db << endl;

    // and query the database
    cout << "Querying the database: " << endl;

    QueryResults ret;
    for (size_t i = 0; i < features.size(); i++)
    {
        db.query(features[i], ret, 4);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.

        cout << "Searching for Image " << i << ". " << ret << endl;
    }

    cout << endl;

    // we can save the database. The created file includes the vocabulary
    // and the entries added
    cout << "Saving database..." << endl;
    db.save("small_db.yml.gz");
    cout << "... done!" << endl;

    // once saved, we can load it again
    cout << "Retrieving database once again..." << endl;
    Database db2("small_db.yml.gz");
    cout << "... done! This is: " << endl
         << db2 << endl;
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    try {
        slamplay::CmdLineParser cml(argc, argv);
        if (cml["-h"] || argc <= 3) {
            cerr << "Usage: " << argv[0] << " <descriptor type> <image_dir> \n\t descriptor types:brisk,surf,orb,akaze" << endl;
            return -1;
        }

        string descriptor = argv[1];
        string dataset_dir = argv[2];

        std::vector<std::string> filenames;
        slamplay::getImageFilenames(dataset_dir, filenames);

        vector<cv::Mat> features = extractFeatures(filenames, descriptor);
        testVocCreation(features);

        testDatabase(features);

    } catch (std::exception &ex) {
        cerr << ex.what() << endl;
    }

    return 0;
}
