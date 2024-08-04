#pragma once

#include <opencv2/opencv.hpp>

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
