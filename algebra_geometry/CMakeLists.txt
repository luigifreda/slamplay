#====================================
# folders to build 

set(FOLDERS 
  eigen
  sophus
  cholmod
)

foreach(folder ${FOLDERS})
  message(STATUS "adding folder: ${folder}")
  add_subdirectory(${folder})
endforeach()
     