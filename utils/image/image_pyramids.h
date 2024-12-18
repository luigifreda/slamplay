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

// from a pyramid (as a vector of images) to a single image where the pyramid images are vertically stacked
inline void composePyrImage(const std::vector<cv::Mat> &pyramid, cv::Mat &out) {
    if (pyramid.empty()) return;

    int rows = 0;
    int cols = 0;
    int posy = 0;

    for (auto &img : pyramid)
    {
        cols = std::max(cols, img.cols);
        rows += img.rows;
    }

    out = cv::Mat(cv::Size(cols, rows), pyramid[0].type(), cv::Scalar::all(0));
    for (size_t ii = 0; ii < pyramid.size(); ii++)
    {
        cv::Rect roi(0, posy, pyramid[ii].cols, pyramid[ii].rows);
        pyramid[ii].copyTo(out(roi));
        posy += pyramid[ii].rows;
    }
}

}  // namespace slamplay