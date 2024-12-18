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
//
// Created by gaoxiang on 19-5-4.
//
#include <gtest/gtest.h>
#include "myslam/common_include.h"
#include "myslam/algorithm.h"

TEST(MyslamTest, Triangulation) {
    Vec3 pt_world(30, 20, 10), pt_world_estimated;
    std::vector<SE3> poses{
            SE3(Eigen::Quaterniond(0, 0, 0, 1), Vec3(0, 0, 0)),
            SE3(Eigen::Quaterniond(0, 0, 0, 1), Vec3(0, -10, 0)),
            SE3(Eigen::Quaterniond(0, 0, 0, 1), Vec3(0, 10, 0)),
    };
    std::vector<Vec3> points;
    for (size_t i = 0; i < poses.size(); ++i) {
        Vec3 pc = poses[i] * pt_world;
        pc /= pc[2];
        points.push_back(pc);
    }

    EXPECT_TRUE(myslam::triangulation(poses, points, pt_world_estimated));
    EXPECT_NEAR(pt_world[0], pt_world_estimated[0], 0.01);
    EXPECT_NEAR(pt_world[1], pt_world_estimated[1], 0.01);
    EXPECT_NEAR(pt_world[2], pt_world_estimated[2], 0.01);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}