#====================================
# folders to build 

set(FOLDERS 
  pointcloud_convert
)

foreach(folder ${FOLDERS})
  message(STATUS "adding folder: ${folder}")
  add_subdirectory(${folder})
endforeach()
     