#include <iostream>
#include <math.h>

#include "pointcloud_utils.h"

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace slamplay;

typedef pcl::PointXYZ PointT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointT> ColorHandlerT;

int main (int argc, char** argv)
{
    pcl::PointCloud<PointT>::Ptr cloud_in (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_transformed (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_aligned (new pcl::PointCloud<PointT>);    

    // Fill in the CloudIn data
    //cloud_in = createRandomCubePointCloud<PointT>();
    cloud_in = createCylinderPointCloud<PointT>();    

    std::cout << "Generated " << cloud_in->size () << " input data points:" << std::endl;
    //for (auto& point : *cloud_in) std::cout << point << std::endl;

    constexpr double theta = M_PI/4; // radiants 
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    // Define a translation of 2.5 meters on the x axis.
    transform.translation() << 0.1, 0.2, 0.1;
    // Rotation matrix: theta radians around axis
    Eigen::AngleAxisf rotation_vector(M_PI / 6, Eigen::Vector3f(1, 1, 1).normalized()); //Rotate 45 degrees around generic unit vector 
    //Eigen::AngleAxisf rotation_vector(theta, Eigen::Vector3f::UnitZ()); //Rotate 45 degrees along the Z axis
    transform.rotate (rotation_vector);
    std::cout << "transform: \n" << transform.matrix() << std::endl;    

    pcl::transformPointCloud (*cloud_in, *cloud_transformed, transform);    

    std::cout << "Transformed " << cloud_in->size () << " data points:" << std::endl;
    //for (auto& point : *cloud_transformed) std::cout << point << std::endl;

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cloud_in);
    icp.setInputTarget(cloud_transformed);

    // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
    //icp.setMaxCorrespondenceDistance (0.05);
    // Set the maximum number of iterations (criterion 1)
    icp.setMaximumIterations (50);
    // Set the transformation epsilon (criterion 2)
    icp.setTransformationEpsilon (1e-8);
    // Set the euclidean distance difference epsilon (criterion 3)
    //icp.setEuclideanFitnessEpsilon (1);
 
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);

    auto icp_transform = icp.getFinalTransformation();
    std::cout << "icp_transform: \n" << icp_transform << std::endl; 

    std::cout << "transform diff: " << (transform.matrix() - icp_transform).norm() << std::endl; 
    
    std::cout << "has converged:" << icp.hasConverged() << std::endl; 
    std::cout << "fitness score: " << icp.getFitnessScore() << std::endl;
    std::cout << icp_transform << std::endl;

    // Transform the input point cloud according to the computed trasform
    pcl::transformPointCloud (*cloud_in, *cloud_aligned, icp_transform);

    // Show alignment
    pcl::visualization::PCLVisualizer viewer("ICP");
    constexpr int point_size = 3; 

    viewer.addCoordinateSystem ();

    viewer.addPointCloud (cloud_in, ColorHandlerT (cloud_in, 0.0, 255.0, 0.0), "cloud_in");
    viewer.addPointCloud (cloud_transformed, ColorHandlerT (cloud_transformed, 255.0, 0.0, 0.0), "cloud_transformed");    
    viewer.addPointCloud (cloud_aligned, ColorHandlerT (cloud_aligned, 0.0, 0.0, 255.0), "cloud_aligned");

    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, "cloud_in");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, "cloud_transformed");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, "cloud_aligned");

    viewer.spin ();

    return (0);
}