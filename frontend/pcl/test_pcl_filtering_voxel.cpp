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
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "macros.h"

std::string dataDir = STR(DATA_DIR); //DATA_DIR set by compilers flag 
std::string pc_file = dataDir + "/pointclouds/table_scene_lms400.pcd";




int main (int argc, char** argv)
{
#if 0    
    pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2 ());
    pcl::PCLPointCloud2::Ptr cloud_filtered (new pcl::PCLPointCloud2 ());

    // Fill in the cloud data
    pcl::PCDReader reader;
    // Replace the path below with the path where you saved your file
    reader.read (pc_file, *cloud); // Remember to download the file first!

    std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height 
              << " data points (" << pcl::getFieldsList (*cloud) << ")." << std::endl;    
#else 

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

    // Fill in the cloud data
    pcl::PCDReader reader;
    // Replace the path below with the path where you saved your file
    reader.read<pcl::PointXYZ> (pc_file, *cloud);

    std::cerr << "Cloud before filtering: " << std::endl;
    std::cerr << *cloud << std::endl;

#endif 

    // Create the filtering object
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud);
    sor.setLeafSize (0.01f, 0.01f, 0.01f);
    sor.filter (*cloud_filtered);

    std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height 
              << " data points (" << pcl::getFieldsList (*cloud_filtered) << ")." << std::endl;

    pcl::PCDWriter writer;
    writer.write ("table_scene_lms400_downsampled.pcd", *cloud_filtered, false);

    using ColorHandlerT = pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>;

    pcl::visualization::PCLVisualizer viewer("statistical outlier removal");
    constexpr int point_size = 3; 

    viewer.addCoordinateSystem ();  

    viewer.addPointCloud (cloud, ColorHandlerT (cloud, 255.0, 0.0, 0.0), "cloud");
    viewer.addPointCloud (cloud_filtered, ColorHandlerT (cloud_filtered, 0.0, 255.0, 0.0), "cloud_filtered");    

    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, "cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, "cloud_filtered");

    viewer.spin ();

  return (0);
}