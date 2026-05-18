#====================================
# folders to build 

set(FOLDERS 
  vslam
  lio_slam
)

foreach(folder ${FOLDERS})
  message(STATUS "adding folder: ${folder}")
  add_subdirectory(${folder})
endforeach()
     