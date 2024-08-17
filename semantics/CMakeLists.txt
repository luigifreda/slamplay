#====================================
# folders to build 

set(FOLDERS 
  sam_tensorrt
)

foreach(folder ${FOLDERS})
  message(STATUS "adding folder: ${folder}")
  add_subdirectory(${folder})
endforeach()