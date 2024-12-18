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
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "macros.h"

std::string dataDir = STR(DATA_DIR); //DATA_DIR set by compilers flag 
std::string pc_file = dataDir + "/pointclouds/table_scene_lms400.pcd";


// https://pcl.readthedocs.io/projects/tutorials/en/latest/statistical_outlier.html#statistical-outlier-removal
int main (int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_inliers (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_outliers (new pcl::PointCloud<pcl::PointXYZ>);

    // Fill in the cloud data
    pcl::PCDReader reader;
    // Replace the path below with the path where you saved your file
    reader.read<pcl::PointXYZ> (pc_file, *cloud);

    std::cerr << "Cloud before filtering: " << std::endl;
    std::cerr << *cloud << std::endl;

    // Create the filtering object
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud);
    sor.setMeanK (50); // The number of neighbors to analyze for each point is set to 50
    sor.setStddevMulThresh (1.0); // the standard deviation multiplier to 1
    sor.filter (*cloud_inliers);

    std::cerr << "Cloud after filtering: " << std::endl;
    std::cerr << *cloud_inliers << std::endl;

#define WRITE_OUTPUT 1
#if WRITE_OUTPUT
    pcl::PCDWriter writer;
    writer.write<pcl::PointXYZ> ("table_scene_lms400_inliers.pcd", *cloud_inliers, false);
#endif 

    sor.setNegative (true);
    sor.filter (*cloud_outliers);
#if WRITE_OUTPUT    
    writer.write<pcl::PointXYZ> ("table_scene_lms400_outliers.pcd", *cloud_outliers, false);
#endif 

    using ColorHandlerT = pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>;

    pcl::visualization::PCLVisualizer viewer("statistical outlier removal");
    constexpr int point_size = 3; 

    viewer.addCoordinateSystem ();  

    viewer.addPointCloud (cloud, ColorHandlerT (cloud, 255.0, 0.0, 0.0), "cloud");
    viewer.addPointCloud (cloud_inliers, ColorHandlerT (cloud_inliers, 0.0, 255.0, 0.0), "cloud_inliers");    

    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, "cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, "cloud_inliers");

    viewer.spin ();

  return (0);
}