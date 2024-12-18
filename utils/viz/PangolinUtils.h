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

#include <Eigen/Dense>

#include <pangolin/pangolin.h>
#include "eigen/EigenUtils.h"

namespace slamplay {

const uint8_t cam_color[3]{250, 0, 26};

inline void render_camera(const Eigen::Matrix4d& T_w_c, float lineWidth = 1.0,
                          const uint8_t* color = cam_color, float sizeFactor = 0.1) {
    const float sz = sizeFactor;
    const float width = 640, height = 480, fx = 500, fy = 500, cx = 320, cy = 240;

    const Eigen::aligned_vector<Eigen::Vector3f> lines = {
        {0, 0, 0},
        {sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz},
        {0, 0, 0},
        {sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
        {0, 0, 0},
        {sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
        {0, 0, 0},
        {sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz},
        {sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz},
        {sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
        {sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
        {sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
        {sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
        {sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz},
        {sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz},
        {sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz}};

    glPushMatrix();
    glMultMatrixd(T_w_c.data());
    glColor3ubv(color);
    glLineWidth(lineWidth);
    pangolin::glDrawLines(lines);
    glPopMatrix();
}

}  // namespace slamplay
