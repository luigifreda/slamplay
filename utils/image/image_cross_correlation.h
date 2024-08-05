#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "image_interpolation.h"

double NCC(const cv::Mat &ref, const cv::Mat &curr,
           const Eigen::Vector2d &pt_ref, const Eigen::Vector2d &pt_curr,
           int ncc_window_size = 3,  // The half-width of the window taken by NCC
           int ncc_area = 49)        // (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); //NCC window area
{
    // zero mean -normalized cross correlation
    // Calculate the mean first
    double mean_ref = 0, mean_curr = 0;
    std::vector<double> values_ref, values_curr;  // The mean of the reference frame and the current frame
    for (int x = -ncc_window_size; x <= ncc_window_size; x++)
        for (int y = -ncc_window_size; y <= ncc_window_size; y++) {
            double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0;
            mean_ref += value_ref;

            double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Eigen::Vector2d(x, y));
            mean_curr += value_curr;

            values_ref.push_back(value_ref);
            values_curr.push_back(value_curr);
        }

    mean_ref /= ncc_area;
    mean_curr /= ncc_area;

    // calculation Zero mean NCC
    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < values_ref.size(); i++) {
        double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
        numerator += n;
        demoniator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
        demoniator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);  // prevent denominator from appearing zero
}