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

#include "eigen/EigenUtils.h"
#include "io/messages.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"

#include <algorithm>
#include <vector>

namespace slamplay {

// adapted from basalt project https://github.com/VladyslavUsenko/basalt
template <typename T>
std::pair<double, Sophus::SE3<T>> alignSVD(const std::vector<int64_t>& filter_t_ns,
                                           const std::vector<Eigen::Matrix<T, 3, 1>>& filter_t_w_i,
                                           const std::vector<int64_t>& gt_t_ns,
                                           std::vector<Eigen::Matrix<T, 3, 1>>& gt_t_w_i) {
    std::vector<Eigen::Matrix<T, 3, 1>> est_associations;
    std::vector<Eigen::Matrix<T, 3, 1>> gt_associations;

    for (size_t i = 0; i < filter_t_w_i.size(); i++) {
        int64_t t_ns = filter_t_ns[i];

        size_t j;
        for (j = 0; j < gt_t_ns.size(); j++) {
            if (gt_t_ns.at(j) > t_ns) break;
        }
        j--;

        if (j >= gt_t_ns.size() - 1) {
            continue;
        }

        double dt_ns = t_ns - gt_t_ns.at(j);
        double int_t_ns = gt_t_ns.at(j + 1) - gt_t_ns.at(j);

        MSG_ASSERT(dt_ns >= 0, "dt_ns " << dt_ns);
        MSG_ASSERT(int_t_ns > 0, "int_t_ns " << int_t_ns);

        // Skip if the interval between gt larger than 100ms
        if (int_t_ns > 1.1e8) continue;

        double ratio = dt_ns / int_t_ns;

        MSG_ASSERT(ratio >= 0, "ratio " << ratio);
        MSG_ASSERT(ratio < 1, "ratio " << ratio);

        Eigen::Matrix<T, 3, 1> gt = (1 - ratio) * gt_t_w_i[j] + ratio * gt_t_w_i[j + 1];

        gt_associations.emplace_back(gt);
        est_associations.emplace_back(filter_t_w_i[i]);
    }

    int num_kfs = est_associations.size();

    Eigen::Matrix<T, 3, Eigen::Dynamic> gt, est;
    gt.setZero(3, num_kfs);
    est.setZero(3, num_kfs);

    for (size_t i = 0; i < est_associations.size(); i++) {
        gt.col(i) = gt_associations[i];
        est.col(i) = est_associations[i];
    }

    Eigen::Matrix<T, 3, 1> mean_gt = gt.rowwise().mean();
    Eigen::Matrix<T, 3, 1> mean_est = est.rowwise().mean();

    gt.colwise() -= mean_gt;
    est.colwise() -= mean_est;

    Eigen::Matrix<T, 3, 3> cov = gt * est.transpose();

    Eigen::JacobiSVD<Eigen::Matrix<T, 3, 3>> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix<T, 3, 3> S;
    S.setIdentity();

    if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0)
        S(2, 2) = -1;

    Eigen::Matrix<T, 3, 3> rot_gt_est = svd.matrixU() * S * svd.matrixV().transpose();
    Eigen::Matrix<T, 3, 1> trans = mean_gt - rot_gt_est * mean_est;

    Sophus::SE3<T> T_gt_est(rot_gt_est, trans);
    Sophus::SE3<T> T_est_gt = T_gt_est.inverse();

    for (size_t i = 0; i < gt_t_w_i.size(); i++) {
        gt_t_w_i[i] = T_est_gt * gt_t_w_i[i];
    }

    double error = 0;
    for (size_t i = 0; i < est_associations.size(); i++) {
        est_associations[i] = T_gt_est * est_associations[i];
        Eigen::Matrix<T, 3, 1> res = est_associations[i] - gt_associations[i];

        error += res.transpose() * res;
    }

    error /= est_associations.size();
    error = std::sqrt(error);
#if 0
    std::cout << "T_align\n"
              << T_gt_est.matrix() << std::endl;
    std::cout << "error " << error << std::endl;
    std::cout << "number of associations " << num_kfs << std::endl;
#endif

    return std::pair<double, Sophus::SE3<T>>(error, T_gt_est);
}

template <typename T>
std::pair<double, Sophus::SE3<T>> alignSVD(const std::vector<int64_t>& filter_t_ns,
                                           const std::vector<Sophus::SE3<T>>& filter_T_w_i,
                                           const std::vector<int64_t>& gt_t_ns,
                                           std::vector<Sophus::SE3<T>>& gt_T_w_i) {
    // Convert the Sophus::SE3 to Eigen::Vector3d (translations only)
    std::vector<Eigen::Matrix<T, 3, 1>> filter_t_w_i;
    std::vector<Eigen::Matrix<T, 3, 1>> gt_t_w_i;

    filter_t_w_i.resize(filter_T_w_i.size());
    gt_t_w_i.resize(gt_T_w_i.size());

    for (size_t i = 0; i < filter_T_w_i.size(); i++) {
        filter_t_w_i[i] = filter_T_w_i[i].translation();
    }

    for (size_t i = 0; i < gt_T_w_i.size(); i++) {
        gt_t_w_i[i] = gt_T_w_i[i].translation();
    }

    // Call the first alignSVD function
    return alignSVD(filter_t_ns, filter_t_w_i, gt_t_ns, gt_t_w_i);
}

}  // namespace slamplay
