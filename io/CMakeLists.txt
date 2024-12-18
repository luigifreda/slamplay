
#====================================
# folders to build 

set(FOLDERS 
  datasets
  fmt
  json
  logs
  time
  yaml
)

foreach(folder ${FOLDERS})
  message(STATUS "adding folder: ${folder}")
  add_subdirectory(${folder})
endforeach()
     