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
#pragma once

#include <opencv2/opencv.hpp>

namespace slamplay {

// Colors
const std::vector<cv::Scalar> CITYSCAPES_COLORS = {
    cv::Scalar(128, 64, 128),
    cv::Scalar(232, 35, 244),
    cv::Scalar(70, 70, 70),
    cv::Scalar(156, 102, 102),
    cv::Scalar(153, 153, 190),
    cv::Scalar(153, 153, 153),
    cv::Scalar(30, 170, 250),
    cv::Scalar(0, 220, 220),
    cv::Scalar(35, 142, 107),
    cv::Scalar(152, 251, 152),
    cv::Scalar(180, 130, 70),
    cv::Scalar(60, 20, 220),
    cv::Scalar(0, 0, 255),
    cv::Scalar(142, 0, 0),
    cv::Scalar(70, 0, 0),
    cv::Scalar(100, 60, 0),
    cv::Scalar(90, 0, 0),
    cv::Scalar(230, 0, 0),
    cv::Scalar(32, 11, 119),
    cv::Scalar(0, 74, 111),
    cv::Scalar(81, 0, 81)};

// Structure to hold clicked point coordinates
struct PointData {
    cv::Point point;
    bool clicked;
};

// Overlay mask on the image
void overlay(cv::Mat& image, cv::Mat& mask, cv::Scalar color = cv::Scalar(128, 64, 128), float alpha = 0.8f, bool showEdge = true) {
    // Draw mask
    cv::Mat ucharMask(image.rows, image.cols, CV_8UC3, color);
    image.copyTo(ucharMask, mask <= 0);
    cv::addWeighted(ucharMask, alpha, image, 1.0 - alpha, 0.0f, image);

    // Draw contour edge
    if (showEdge)
    {
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(mask <= 0, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
        cv::drawContours(image, contours, -1, cv::Scalar(255, 255, 255), 2);
    }
}

// Function to handle mouse events
void onMouse(int event, int x, int y, int flags, void* userdata) {
    PointData* pd = (PointData*)userdata;
    if (event == cv::EVENT_LBUTTONDOWN) {
        // Save the clicked coordinates
        pd->point = cv::Point(x, y);
        pd->clicked = true;
    }
}

}  // namespace slamplay