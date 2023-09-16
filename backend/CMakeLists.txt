#====================================
# folders to build 

set(FOLDERS 
  g2o  
  ceres
  gtsam
  eigen_optimization
  sesync
  slam_ba_simple_g2o_ceres
  slam_pose_graph_simple_g2o
)

foreach(folder ${FOLDERS})
  message(STATUS "adding folder: ${folder}")
  add_subdirectory(${folder})
endforeach()
     