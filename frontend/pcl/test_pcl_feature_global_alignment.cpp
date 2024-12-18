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
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "macros.h"

std::string dataDir = STR(DATA_DIR); //DATA_DIR set by compilers flag 
std::string pc1_file = dataDir + "/pointclouds/toalign/chef.pcd";
std::string pc2_file = dataDir + "/pointclouds/toalign/rs1.pcd";

// Types
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

// Align a rigid object to a scene with clutter and occlusions
// 1) Downsample via grid filtering 
// 2) Estimate normals 
// 3) Extract features 
// 4) 
int main (int argc, char **argv)
{
    // Point clouds
    PointCloudT::Ptr object (new PointCloudT);
    PointCloudT::Ptr object_aligned (new PointCloudT);
    PointCloudT::Ptr scene (new PointCloudT);
    FeatureCloudT::Ptr object_features (new FeatureCloudT);
    FeatureCloudT::Ptr scene_features (new FeatureCloudT);

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
    
    // Downsample
    pcl::console::print_highlight ("Downsampling...\n");
    pcl::VoxelGrid<PointNT> grid;
    const float leaf_size = 0.005f;
    grid.setLeafSize (leaf_size, leaf_size, leaf_size);
    grid.setInputCloud (object);
    grid.filter (*object);
    grid.setInputCloud (scene);
    grid.filter (*scene);
    
#define USE_OMP 0    
    // Estimate normals for scene
    pcl::console::print_highlight ("Estimating scene normals...\n");
#if USE_OMP
    pcl::NormalEstimationOMP<PointNT,PointNT> nest;
#else 
    pcl::NormalEstimation<PointNT,PointNT> nest;
#endif     
    nest.setRadiusSearch (0.01);
    nest.setInputCloud (scene);
    nest.compute (*scene);
    
    // Estimate features
    pcl::console::print_highlight ("Estimating features...\n");
    FeatureEstimationT fest;
    fest.setRadiusSearch (0.025);
    fest.setInputCloud (object);
    fest.setInputNormals (object);
    fest.compute (*object_features);
    fest.setInputCloud (scene);
    fest.setInputNormals (scene);
    fest.compute (*scene_features);
    
    // Perform alignment
    pcl::console::print_highlight ("Starting alignment...\n");
    pcl::SampleConsensusPrerejective<PointNT,PointNT,FeatureT> align;
    align.setInputSource (object);
    align.setSourceFeatures (object_features);
    align.setInputTarget (scene);
    align.setTargetFeatures (scene_features);
    align.setMaximumIterations (50000);    // Number of RANSAC iterations
    align.setNumberOfSamples (3);          // Number of points to sample for generating/prerejecting a pose
    align.setCorrespondenceRandomness (5); // Number of nearest features to use
    align.setSimilarityThreshold (0.9f);   // Polygonal edge length similarity threshold
    align.setMaxCorrespondenceDistance (2.5f * leaf_size); // Inlier threshold
    align.setInlierFraction (0.25f);       // Required inlier fraction for accepting a pose hypothesis
    {
        pcl::ScopeTime t("Alignment");
        align.align (*object_aligned);
    }
    
    if (align.hasConverged ())
    {
        // Print results
        printf ("\n");
        Eigen::Matrix4f transformation = align.getFinalTransformation ();
        pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
        pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
        pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
        pcl::console::print_info ("\n");
        pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
        pcl::console::print_info ("\n");
        pcl::console::print_info ("Inliers: %i/%i\n", align.getInliers ().size (), object->size ());
        
        // Show alignment
        pcl::visualization::PCLVisualizer viewer("Global Alignment");
        viewer.addPointCloud (scene, ColorHandlerT (scene, 0.0, 255.0, 0.0), "scene");
        viewer.addPointCloud (object_aligned, ColorHandlerT (object_aligned, 0.0, 0.0, 255.0), "object_aligned");
        viewer.spin ();
    }
    else
    {
        pcl::console::print_error ("Alignment failed!\n");
        return (1);
    }
    
    return (0);
}