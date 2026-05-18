#====================================
# folders to build 

set(FOLDERS 
  camera_model  
  depth_dl
  features
  features_dl
  motion_estimation_2d_2d  
  motion_estimation_3d_2d  
  motion_estimation_3d_3d
  opencv  
  pointclouds
  stereo_vision
  triangulation
  pcl
  icp 
  opticflow_direct_method
  trajectory_alignment
  nearest_neighbor
  linear_fitting
  laser_2d
  laser_3d
  gnss_navigation
  imu
)

foreach(folder ${FOLDERS})
  message(STATUS "adding folder: ${folder}")
  add_subdirectory(${folder})
endforeach()
     