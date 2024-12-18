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
#include <chrono>
#include <string>
#include "macros.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

std::string dataDir = STR(DATA_DIR); //DATA_DIR set by compilers flag 
string image_file = dataDir + "/ubuntu.png";

int main(int argc, char **argv) 
{
    //Read the image specified by argv[1]
    cv::Mat image;
    image = cv::imread(image_file);//cv::imread function reads the image under the specified path

    //Determine whether the image file is read correctly
    if (image.data == nullptr) {//The data does not exist, it may be that the file does not exist
      cerr << "document" << image_file << "does not exist." << endl;
      return 0;
    }

    //The file is successfully read, first output some basic information
    cout << "Image width is " << image.cols << ", height is " << image.rows << ", the number of channels is " << image.channels() << endl;
    cv::imshow("image", image);//display image with cv::imshow
    cv::waitKey(0);//Pause the program, waiting for a key input

    //Determine the type of image
    if (image.type() != CV_8UC1 && image.type() != CV_8UC3) {
    //The image type does not meet the requirements
      cout << "Please enter a color or grayscale image." << endl;
      return 0;
    }

    //Traverse the image, please note that the following traversal methods can also be used for random pixel access
    //Use std::chrono to time the algorithm
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for (size_t y = 0; y < image.rows; y++) {
      //Use cv::Mat::ptr to get the row pointer of the image
      unsigned char *row_ptr = image.ptr<unsigned char>(y);//row_ptr is the head pointer of row y
      for (size_t x = 0; x < image.cols; x++) {
        //Access the pixel at x,y
        unsigned char *data_ptr = &row_ptr[x * image.channels()];//data_ptr points to the pixel data to be accessed
        //Output each channel of the pixel, if it is a grayscale image, there is only one channel
        for (int c = 0; c != image.channels(); c++) {
          unsigned char data = data_ptr[c];//data is the value of the cth channel of I(x,y)
        }
      }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast < chrono::duration < double >> (t2 - t1);
    cout << "When traversing the image: " << time_used.count() << " seconds" << endl;

    //Copy about cv::Mat
    //Direct assignment does not copy data
    cv::Mat image_another = image;
    //Modifying image_another will cause the image to change
    image_another(cv::Rect(0, 0, 100, 100)).setTo(0);//Set the 100*100 block in the upper left corner to zero
    cv::imshow("image", image);
    cv::waitKey(0);

    //Use the clone function to copy the data
    cv::Mat image_clone = image.clone();
    image_clone(cv::Rect(0, 0, 100, 100)).setTo(255);
    cv::imshow("image", image);
    cv::imshow("image_clone", image_clone);
    cv::waitKey(0);

    //There are many basic operations on images, such as cutting, rotating, scaling, etc., which are not introduced one by one due to space limitations. Please refer to the OpenCV official documentation to query the calling method of each function.
    cv::destroyAllWindows();
    return 0;
}
