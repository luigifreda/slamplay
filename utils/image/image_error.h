#pragma once 

#include <opencv2/opencv.hpp>

#include "String.h"

inline void plotImageErrorWithColorbar(cv::Mat& error)
{
    const int font = cv::FONT_HERSHEY_SIMPLEX; // Set font type
    const double scale = 0.5; // Set font scale
    const double scale_stroke = 0.3; // Set font scale    
    const int thickness = 1; // Set font thickness
    const int thickness_stroke = 1; // Set font thickness    
    const int bar_width = 20; // Set colorbar width    
    const int bar_left_offset = 50; // Set colorbar x left offset 

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(error, &minVal, &maxVal, &minLoc, &maxLoc); // Get min and max

    cv::Mat error_norm, error_color;
    cv::normalize(error, error_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1); // Scale to [0, 255] range
    cv::cvtColor(error_norm, error_color, cv::COLOR_GRAY2BGR); // Convert to BGR color space

    // Assume error_color is a color image of image errors
    int width = error_color.cols; // Get image width
    int height = error_color.rows; // Get image height
    int bar_height = height / 2; // Set colorbar height
    int bar_x = width - bar_width - bar_left_offset; // Set colorbar x position
    int bar_y = height / 4; // Set colorbar y position

    // Draw colorbar segments with different colors
    for (int i = 0; i < bar_height; i++) {
        int value = 255 - i * 255 / bar_height; // Map i to [255, 0] range
        cv::Scalar color(value, value, value); // Create grayscale color
        cv::line(error_color, cv::Point(bar_x, bar_y + i), cv::Point(bar_x + bar_width, bar_y + i), color, 1); // Draw horizontal line with color
    }

    // Draw colorbar labels with min and max values
    std::string min_val_str = to_string_with_precision(minVal,2);
    std::string max_val_str = to_string_with_precision(maxVal,2);

    cv::putText(error_color, min_val_str, cv::Point(bar_x + bar_width + 5, bar_y + bar_height), font, scale, cv::Scalar(0, 0, 0), thickness); // Draw min value at bottom
    cv::putText(error_color, min_val_str, cv::Point(bar_x + bar_width + 5, bar_y + bar_height), font, scale_stroke, cv::Scalar(255, 255, 255), thickness_stroke); // Draw min value at bottom

    cv::putText(error_color, max_val_str, cv::Point(bar_x + bar_width + 5, bar_y), font, scale, cv::Scalar(0, 0, 0), thickness); // Draw max value at top
    cv::putText(error_color, max_val_str, cv::Point(bar_x + bar_width + 5, bar_y), font, scale_stroke, cv::Scalar(255, 255, 255), thickness_stroke); // Draw max value at top    

    cv::imshow("Error color map with legend", error_color); // Show color image with legend
}