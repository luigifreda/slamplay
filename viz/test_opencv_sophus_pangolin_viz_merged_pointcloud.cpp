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
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>  //for formatting strings
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>

#include "macros.h"

using namespace std;

std::string dataDir = STR(DATA_DIR); //DATA_DIR set by compilers flag 
string pose_file = dataDir + "/rgbd/pose.txt";
string color_file_prefix = dataDir + "/rgbd/color";
string depth_file_prefix = dataDir + "/rgbd/depth";

using TrajectoryType = vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using PointCloud6D = vector<Vector6d, Eigen::aligned_allocator<Vector6d>>;

//Drawing in pangolin, already written, no adjustments needed
void showPointCloud(const PointCloud6D &pointcloud);

int main(int argc, char **argv) 
{
    vector<cv::Mat> colorImgs, depthImgs;//color map and depth map
    TrajectoryType poses;//camera pose

    ifstream fin(pose_file);
    if (!fin) {
        cerr << "Please run this program in the directory with pose.txt" << endl;
        return 1;
    }

    for (int i = 0; i < 5; i++) {
        boost::format fmt("%s/%d.%s");//image file format
        colorImgs.push_back(cv::imread((fmt % color_file_prefix % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % depth_file_prefix % (i + 1) % "pgm").str(), -1));//use -1 to read the original image

        double data[7] = {0};
        for (auto &d:data) fin >> d;
        Sophus::SE3d pose(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                          Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(pose);
    }

    //Calculate point cloud and stitch
    //internal parameters of the camera
    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthScale = 1000.0;

    PointCloud6D pointcloud;
    pointcloud.reserve(1000000);

    for (int i = 0; i < 5; i++) 
    {
        cout << "Convert image in: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Sophus::SE3d T = poses[i];
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u];//depth value
                if (d == 0) continue;//0 means not measured
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;

                Vector6d p;
                p.head<3>() = pointWorld;
                p[5] = color.data[v * color.step + u * color.channels()];    //blue
                p[4] = color.data[v * color.step + u * color.channels() + 1];//green
                p[3] = color.data[v * color.step + u * color.channels() + 2];//red
                pointcloud.push_back(p);
            }
    }

    cout << "Shared point cloud " << pointcloud.size() << " point." << endl;
    showPointCloud(pointcloud);
    return 0;
}

void showPointCloud(const PointCloud6D &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);//sleep 5 ms
    }
    return;
}
