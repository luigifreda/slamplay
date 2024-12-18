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
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "macros.h"

using namespace std;
using namespace cv;


struct OpticFlowParams
{
    // forward or inverse compositional 
    bool inverse = true; 

    // patch 
    int half_patch_size = 4;
    
    // optimization 
    int num_iterations = 10;
    double eps_convergence = 1e-2;

    // multi-level 
    int num_pyramid_levels = 4;
    double pyramid_scale = 0.5;
};

/// Optical flow tracker and interface
class OpticalFlowTracker 
{
public:
    OpticalFlowTracker(
        const Mat &img1_,
        const Mat &img2_,
        const vector<KeyPoint> &kp1_,
        vector<KeyPoint> &kp2_,
        vector<bool> &success_,
        OpticFlowParams &params_, 
        bool has_initial_ = false) :
        img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), params(params_),
        has_initial(has_initial_) {}

    void calculateOpticalFlow(const Range &range);

private:
    const Mat &img1;
    const Mat &img2;
    const vector<KeyPoint> &kp1;
    vector<KeyPoint> &kp2;
    vector<bool> &success;
    OpticFlowParams& params; 
    bool has_initial = false;

};

/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
inline void opticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    OpticFlowParams &params, 
    bool has_initial = false) 
{
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, params, has_initial);
    cv::parallel_for_(Range(0, kp1.size()),
                  std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));
}

/**
 * multi level optical flow, scale of pyramid is set to 2 by default
 * the image pyramid will be create inside the function
 * @param [in] img1 the first pyramid
 * @param [in] img2 the second pyramid
 * @param [in] kp1 keypoints in img1
 * @param [out] kp2 keypoints in img2
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse set true to enable inverse formulation
 */
inline void opticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    OpticFlowParams &params) 
{
    // parameters
    const int& num_pyramid_levels = params.num_pyramid_levels;
    const double& pyramid_scale = params.pyramid_scale;

    vector<double> scales; 

    // create pyramids
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    vector<Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < num_pyramid_levels; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
            scales.push_back(1.0);
        } else {
            Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
            scales.push_back(scales[i-1] * pyramid_scale);
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "build pyramid time: " << time_used.count() << endl;

    // coarse-to-fine LK tracking in pyramids
    vector<KeyPoint> kp1_pyr, kp2_pyr;
    for (auto &kp:kp1) {
        auto kp_top = kp;
        kp_top.pt *= scales[num_pyramid_levels - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }

    for (int level = num_pyramid_levels - 1; level >= 0; level--) {
        // from coarse to fine
        success.clear();
        t1 = chrono::steady_clock::now();
        opticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, params, true);
        t2 = chrono::steady_clock::now();
        auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << "track pyr " << level << " cost time: " << time_used.count() << endl;

        if (level > 0) {
            for (auto &kp: kp1_pyr)
                kp.pt /= pyramid_scale;
            for (auto &kp: kp2_pyr)
                kp.pt /= pyramid_scale;
        }
    }

    for (auto &kp: kp2_pyr)
        kp2.push_back(kp);
}


/**
 * get a gray scale value from reference image (bi-linear interpolated)
 * @param img
 * @param x
 * @param y
 * @return the interpolated value of this pixel
 */
inline float getPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2; // -2 to get a bilinear interpolation 
    if (y >= img.rows - 1) y = img.rows - 2;
    
    const float xx = x - floor(x);
    const float yy = y - floor(y);
    const int x_a1 = std::min(img.cols - 1, int(x) + 1);
    const int y_a1 = std::min(img.rows - 1, int(y) + 1);
    
    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
    + xx * (1 - yy) * img.at<uchar>(y, x_a1)
    + (1 - xx) * yy * img.at<uchar>(y_a1, x)
    + xx * yy * img.at<uchar>(y_a1, x_a1);
}


inline void OpticalFlowTracker::calculateOpticalFlow(const Range &range) 
{
    // parameters
    const int& half_patch_size = params.half_patch_size;
    const int& num_iterations = params.num_iterations;
    const double& eps_convergence = params.eps_convergence;
    const bool& inverse = params.inverse;

    for (int i = range.start; i < range.end; i++) 
    {
        const auto& kp = kp1[i];
        double dx = 0, dy = 0; // dx,dy need to be estimated
        if (has_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();    // hessian
        Eigen::Vector2d b = Eigen::Vector2d::Zero();    // bias
        Eigen::Vector2d J;  // jacobian
        for (int iter = 0; iter < num_iterations; iter++) 
        {
            if (inverse == false) {
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            } else {
                // only reset b
                b = Eigen::Vector2d::Zero();
            }

            cost = 0;

            // compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) 
                {
                    // Problem: we are interatively minimizing f(u,v) via Gauss-Newton 
                    // (see the paper "Lucas-Kanade 20 Years On: A Unifying Framework" by Baker and Mattheus)
                    // NORMAL MODE: 
                    //  f(u,v) = ||I1(x,y) - I2(x+dx+u,y+dy+v)||^2 = ||e(u,v)||^2     
                    // INVERSE COMPOSITIONAL MODE: 
                    //  f(u,v) = ||I1(x-u,y-v) - I2(x+dx,y+dy)||^2 = ||e(u,v)||^2
                    //    This comes from minimizing: f(u,v) = ||I1(W(x;-u,-v)) - I2(W(x;dx,dv))||^2  (see equation (31) in the paper)
                    //    With dP=(-u,-v) and p=(dx,dv) => The update is W(x;p) = W(x;p)°W(x;dp)^-1 => (dx,dy)+=(u,v)
                    // With both approaches, at each iteration, we find (u,v) and then update:
                    // (dx,dy)+=(u,v)   
                    // 
                    // f(u,v) = e(u,v)^T * e(u,v)  with e(u,v) \in IR
                    // e(u,v) ~= e(0,0) + J * [u,v]^T  where J = de/d(u,v) \in IR^1x2  and [u,v]^T \in IR^2
                    // NOTE: here J is a row. Below, J is a column so the transpose operations are managed in a different way. 
                    // e(u,v) ~= e(0,0)^T*e(0,0) + 2*e(0,0)^T*J*[u,v]^T + [u,v]*J^T*J*[u,v]^T
                    // and by optimizing for (u,v) (that is imposing the derivative of the quadratic approx is equal to zero) 
                    // we get the normal classic equations 
                    // J^T*J*[u,v]^T = -J^T*e
                     
                    double error = getPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                   getPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);
                    // Jacobian
                    // We apply Gauss-Netwon method and approximate 
                    // NORMAL MODE: 
                    //  I2(x+dx+u,y+dy+v) ~= I2(x+dx,y+dy) + dI2(x+dx,y+dy)/dx * u + dI2(x+dx,y+dy)/dy * v
                    //  That means: J = de/d(u,v) = -[dI2(x+dx,y+dy)/dx, dI2(x+dx,y+dy)/dy]^T
                    //  By using the central difference one has: 
                    //  dI2(x+dx,y+dy)/dx ~= (I2(x+dx+h,y+dy) - I2(x+dx-h,y+dy))/2*h 
                    //  and with h=1 => dI2(x+dx,y+dy)/dx ~= 0.5*(I2(x+dx+1,y+dy) - I2(x+dx-1,y+dy))
                    //  In the same way, dI2(x+dx,y+dy)/dy ~= (I2(x+dx,y+dy+h) - I2(x+dx,y+dy-h))/2*h 
                    //  and with h=1 => dI2(x+dx,y+dy)/dy ~= 0.5*(I2(x+dx,y+dy+1) - I2(x+dx,y+dy-1))
                    // INVERSE COMPOSITIONAL MODE: 
                    //  I1(x-u,y-v) ~= I1(x,y) - dI1(x,y)/dx * u - dI1(x,y)/dy * v
                    //  That entails: J = de/d(u,v) = [-dI1(x,y)/dx, -dI1(x,y)/dy]^T
                    //  We compute the derivatives dI1(x,y)/dx and dI1(x,y)/dy as above by using the central differences 
                    if (inverse == false) {
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (getPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                   getPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                            0.5 * (getPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                   getPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
                        );
                    } else if (iter == 0) {
                        // In inverse compositional mode, J keeps same for all iterations
                        // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (getPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                   getPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                            0.5 * (getPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                   getPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))
                        );
                    }
                    // compute H, b and set cost;
                    b += -error * J;
                    cost += error * error;
                    if (inverse == false || iter == 0) {
                        // also update H
                        H += J * J.transpose();
                    }
                }

            // compute update
            Eigen::Vector2d update = H.ldlt().solve(b);

            if (std::isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }

            if (iter > 0 && cost > lastCost) {
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            if (update.norm() < eps_convergence) {
                // converge
                break;
            }
        }

        success[i] = succ;

        // set kp2
        kp2[i].pt = kp.pt + Point2f(dx, dy);
    }
}
