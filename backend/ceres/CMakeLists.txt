include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

#file(GLOB_RECURSE SOURCES "./*.cpp")
file(GLOB SOURCES "./*.cpp")  # in this folder we do not recursively search in the lib folders

# Add an executable target for each source file
foreach(source_file ${SOURCES})
  message(STATUS "compiling ${source_file}")
  get_filename_component(target_name ${source_file} NAME_WE)
  add_executable(${target_name} ${source_file})
  target_link_libraries(${target_name} ${CORELIBS})
endforeach()


include_directories(${CMAKE_CURRENT_SOURCE_DIR}) # for including "common" folder with utils 
add_subdirectory(pose_graph_2d)
add_subdirectory(pose_graph_3d)