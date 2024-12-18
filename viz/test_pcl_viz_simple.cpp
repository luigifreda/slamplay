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
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "macros.h"

std::string dataDir = STR(DATA_DIR); //DATA_DIR set by compilers flag 
std::string pc1_file = dataDir + "/pointclouds/toalign/chef.pcd";
std::string pc2_file = dataDir + "/pointclouds/toalign/rs1.pcd";

// Types
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

// Align a rigid object to a scene with clutter and occlusions
int main (int argc, char **argv)
{
    // Point clouds
    PointCloudT::Ptr object (new PointCloudT);
    PointCloudT::Ptr scene (new PointCloudT);

    // Get input object and scene
    if (argc != 3)
    {
        pcl::console::print_info ("Syntax is: %s object.pcd scene.pcd\n", argv[0]);
    }
    else
    {
        pc1_file = argv[1];
        pc2_file = argv[2];        
    }
    
    // Load object and scene
    pcl::console::print_highlight ("Loading point clouds %s and %s \n",pc1_file.c_str(),pc2_file.c_str());
    if (pcl::io::loadPCDFile<PointNT> (pc1_file, *object) < 0 ||
        pcl::io::loadPCDFile<PointNT> (pc2_file, *scene) < 0)
    {
        pcl::console::print_error ("Error loading object/scene file!\n");
        return (1);
    }
    
    // Show alignment
    pcl::visualization::PCLVisualizer viewer("Point Clouds");
    viewer.addCoordinateSystem ();    
    viewer.addPointCloud (scene, ColorHandlerT (scene, 0.0, 255.0, 0.0), "scene");
    viewer.addPointCloud (object, ColorHandlerT (object, 0.0, 0.0, 255.0), "object");
    //viewer.setPosition(800, 400); // Setting visualiser window position    
    viewer.spin ();
    
    return (0);
}