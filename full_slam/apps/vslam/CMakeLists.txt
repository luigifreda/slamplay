# add_executable(run_vslam_kitti_stereo run_vslam_kitti_stereo.cpp)
# target_link_libraries(run_vslam_kitti_stereo PRIVATE vslam)

# add_executable(run_vslam_euroc_stereo run_vslam_euroc_stereo.cpp)
# target_link_libraries(run_vslam_euroc_stereo PRIVATE vslam)


include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

file(GLOB_RECURSE SOURCES "./*.cpp")

# Add an executable target for each source file
foreach(source_file ${SOURCES})
  message(STATUS "compiling ${source_file}")
  get_filename_component(target_name ${source_file} NAME_WE)
  add_executable(${target_name} ${source_file})
  target_link_libraries(${target_name} vslam ${CORELIBS})
endforeach()