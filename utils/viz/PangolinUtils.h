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
