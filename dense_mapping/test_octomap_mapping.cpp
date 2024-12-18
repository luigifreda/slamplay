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

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <octomap/octomap.h>    //for octomap

#include <Eigen/Geometry>
#include <boost/format.hpp>  //for formatting strings

#include "macros.h"

std::string dataDir = STR(DATA_DIR); // DATA_DIR set by compilers flag 

int main(int argc, char **argv) 
{
    string dataset_dir = dataDir + "/dense_mapping";
    if( argc== 2) {
        dataset_dir = argv[1];
    } else {
      cout << "usage: " << argv[0] <<" <dataset dir>" << endl;
    }
    
    vector<cv::Mat> colorImgs, depthImgs;//color map and depth map
    vector<Eigen::Isometry3d> poses;//camera pose

    ifstream fin(dataset_dir + "/pose.txt");
    if (!fin) {
        cerr << "cannot find pose file" << endl;
        return 1;
    }

    for (int i = 0; i < 5; i++) {
        boost::format fmt(dataset_dir + "/%s/%d.%s");//image file format
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "png").str(), -1));//use -1 to read the original image

        double data[7] = {0};
        for (int i = 0; i < 7; i++) {
            fin >> data[i];
        }
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(T);
    }

    //Calculate point cloud and stitch
    //internal parameters of the camera
    double cx = 319.5;
    double cy = 239.5;
    double fx = 481.2;
    double fy = -480.0;
    double depthScale = 5000.0;

    cout << "Converting image to Octomap ..." << endl;

    //octomap tree
    octomap::OcTree tree(0.01);//parameter is resolution

    for (int i = 0; i < 5; i++) {
        cout << "convert image in: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];

        octomap::Pointcloud cloud;//the point cloud in octomap

        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u];//depth value
                if (d == 0) continue;//0 means not measured
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;
                //Put the points in the world coordinate system into the point cloud
                cloud.push_back(pointWorld[0], pointWorld[1], pointWorld[2]);
            }

        //Store the point cloud into the octree map, given the origin, so that the projection line can be calculated
        tree.insertPointCloud(cloud, octomap::point3d(T(0, 3), T(1, 3), T(2, 3)));
    }

    //Update the occupancy information of the intermediate node and write it to disk
    tree.updateInnerOccupancy();
    cout << "saving octomap ... " << endl;
    tree.writeBinary("octomap.bt");
    return 0;
}
