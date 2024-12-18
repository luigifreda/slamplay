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
#include <opencv2/opencv_modules.hpp>

#ifdef HAVE_OPENCV_CONTRIB

#include <opencv2/line_descriptor.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include "macros.h"

std::string dataDir = STR(DATA_DIR); // DATA_DIR set by compilers flag

#define MATCHES_DIST_THRESHOLD 25

using namespace cv;
using namespace cv::line_descriptor;

static void help()
{
    std::cout << "\nThis example shows the functionalities of lines extraction "
              << "and descriptors computation furnished by BinaryDescriptor class\n"
              << "Please, run this sample using a command in the form\n"
              << "./test_line_matching <path_to_input_image 1> <path_to_input_image 2>" << std::endl;
}

int main(int argc, char **argv)
{
#if 1    
    std::string img1_file = dataDir + "/lines/cameraman.jpg";
    std::string img2_file = dataDir + "/lines/cameraman_transformed.jpg";
#else 
    std::string img1_file = dataDir + "/rgbd2/1.png";
    std::string img2_file = dataDir + "/rgbd2/2.png";
#endif     

    if (argc == 3)
    {
        img1_file = argv[1];
        img2_file = argv[2];
    }
    else if (argc < 3)
    {
        help();
    }

    /* load image */
    cv::Mat imageMat1 = imread(img1_file, 1);
    cv::Mat imageMat2 = imread(img2_file, 1);

    if (imageMat1.data == NULL || imageMat2.data == NULL)
    {
        std::cout << "Error, images could not be loaded. Please, check their path" << std::endl;
    }

    /* create binary masks */
    cv::Mat mask1 = Mat::ones(imageMat1.size(), CV_8UC1);
    cv::Mat mask2 = Mat::ones(imageMat2.size(), CV_8UC1);

    /* create a pointer to a BinaryDescriptor object with default parameters */
    Ptr<BinaryDescriptor> bd = BinaryDescriptor::createBinaryDescriptor();

    /* compute lines and descriptors */
    std::vector<KeyLine> keylines1, keylines2;
    cv::Mat descr1, descr2;

    (*bd)(imageMat1, mask1, keylines1, descr1, false, false);
    (*bd)(imageMat2, mask2, keylines2, descr2, false, false);

    /* select keylines from first octave and their descriptors */
    std::vector<KeyLine> lbd_octave1, lbd_octave2;
    Mat left_lbd, right_lbd;
    for (int i = 0; i < (int)keylines1.size(); i++)
    {
        if (keylines1[i].octave == 0)
        {
            lbd_octave1.push_back(keylines1[i]);
            left_lbd.push_back(descr1.row(i));
        }
    }

    for (int j = 0; j < (int)keylines2.size(); j++)
    {
        if (keylines2[j].octave == 0)
        {
            lbd_octave2.push_back(keylines2[j]);
            right_lbd.push_back(descr2.row(j));
        }
    }

    /* create a BinaryDescriptorMatcher object */
    Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

    /* require match */
    std::vector<DMatch> matches;
    bdm->match(left_lbd, right_lbd, matches);

    /* select best matches */
    std::vector<DMatch> good_matches;
    for (int i = 0; i < (int)matches.size(); i++)
    {
        if (matches[i].distance < MATCHES_DIST_THRESHOLD)
            good_matches.push_back(matches[i]);
    }

    /* plot matches */
    cv::Mat outImg;
    cv::Mat scaled1, scaled2;
    std::vector<char> mask(matches.size(), 1);
    drawLineMatches(imageMat1, lbd_octave1, imageMat2, lbd_octave2, good_matches, outImg, Scalar::all(-1), Scalar::all(-1), mask,
                    DrawLinesMatchesFlags::DEFAULT);

    namedWindow("BinaryDescriptor Matches", WINDOW_NORMAL );// Create a window for display.
    imshow("BinaryDescriptor Matches", outImg);

    /* create an LSD detector */
    Ptr<LSDDetector> lsd = LSDDetector::createLSDDetector();

    /* detect lines */
    std::vector<KeyLine> klsd1, klsd2;
    Mat lsd_descr1, lsd_descr2;
    lsd->detect(imageMat1, klsd1, 2, 2, mask1);
    lsd->detect(imageMat2, klsd2, 2, 2, mask2);

    /* compute descriptors for lines from first octave */
    bd->compute(imageMat1, klsd1, lsd_descr1);
    bd->compute(imageMat2, klsd2, lsd_descr2);

    /* select lines and descriptors from first octave */
    std::vector<KeyLine> octave0_1, octave0_2;
    Mat leftDEscr, rightDescr;
    for (int i = 0; i < (int)klsd1.size(); i++)
    {
        if (klsd1[i].octave == 0)
        {
            octave0_1.push_back(klsd1[i]);
            leftDEscr.push_back(lsd_descr1.row(i));
        }
    }

    for (int j = 0; j < (int)klsd2.size(); j++)
    {
        if (klsd2[j].octave == 0)
        {
            octave0_2.push_back(klsd2[j]);
            rightDescr.push_back(lsd_descr2.row(j));
        }
    }

    /* compute matches */
    std::vector<DMatch> lsd_matches;
    bdm->match(leftDEscr, rightDescr, lsd_matches);

    /* select best matches */
    good_matches.clear();
    for (int i = 0; i < (int)lsd_matches.size(); i++)
    {
        if (lsd_matches[i].distance < MATCHES_DIST_THRESHOLD)
            good_matches.push_back(lsd_matches[i]);
    }

    /* plot matches */
    cv::Mat lsd_outImg;
    //resize(imageMat1, imageMat1, Size(imageMat1.cols / 2, imageMat1.rows / 2), 0, 0, INTER_LINEAR_EXACT);
    //resize(imageMat2, imageMat2, Size(imageMat2.cols / 2, imageMat2.rows / 2), 0, 0, INTER_LINEAR_EXACT);
    std::vector<char> lsd_mask(matches.size(), 1);
    drawLineMatches(imageMat1, octave0_1, imageMat2, octave0_2, good_matches, lsd_outImg, Scalar::all(-1), Scalar::all(-1), lsd_mask,
                    DrawLinesMatchesFlags::DEFAULT);

    namedWindow("LSD matches", WINDOW_NORMAL );// Create a window for display.
    imshow("LSD matches", lsd_outImg);
    waitKey();
}

#else

int main()
{
    std::cerr << "OpenCV was built without features2d module" << std::endl;
    return 0;
}

#endif // HAVE_OPENCV_CONTRIB