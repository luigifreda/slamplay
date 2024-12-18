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
#include "SamInterface.h"

#include "io/file_utils.h"
#include "sam_tensorrt/Sam.h"
#include "sam_tensorrt/sam_export.h"
#include "sam_tensorrt/sam_utils.h"

#include <macros.h>

const std::string kDataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag
const std::string kSamDataDir = kDataDir + "/segment_anything";

///////////////////////////////////////////////////////////////////////////////
using namespace std;
using namespace cv;

int main(int argc, char const* argv[]) {
    const std::string inputImage = kSamDataDir + "/truck-vga.png";

    cv::Mat frame = cv::imread(inputImage);
    std::cout << "frame size:" << frame.size << std::endl;

    SamInterface samIf;
    samIf.loadModels();

    samIf.processEmbedding(frame);
    cv::Mat mask = samIf.processAutoSegment(frame);

    cv::Mat outImage;
    samIf.showAutoSegmentResult(frame, mask, outImage);
    cv::imshow("auto segmentation", outImage);

    cv::waitKey(0);
}