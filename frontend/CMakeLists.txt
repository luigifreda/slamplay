#====================================
# folders to build 

set(FOLDERS 
  camera_model  
  features
  features_tensorrt
  motion_estimation_2D_2D  
  motion_estimation_3D_2D  
  motion_estimation_3D_3D
  opencv  
  pointclouds
  stereo_vision
  triangulation
  pcl
  icp 
  opticflow_direct_method
)

foreach(folder ${FOLDERS})
  message(STATUS "adding folder: ${folder}")
  add_subdirectory(${folder})
endforeach()
     