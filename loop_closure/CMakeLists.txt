include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

file(GLOB_RECURSE SOURCES "./*.cpp")
#file(GLOB SOURCES "./*.cpp")  # in this folder we do not recursively search in the lib folders

message(STATUS "SOURCES: ${SOURCES}")
list(FILTER SOURCES EXCLUDE REGEX ".*hfnet\\.cpp$") # filter our test related to hfnet (it may include tensorflow which may use a different Eigen version)

# Add an executable target for each source file
foreach(source_file ${SOURCES})
  message(STATUS "compiling ${source_file}")
  get_filename_component(target_name ${source_file} NAME_WE)
  add_executable(${target_name} ${source_file})
  target_link_libraries(${target_name} ${CORELIBS})
endforeach()


# Treat separately potential users of tensorflow (e.g. hfnet_features)
set(HFNET_TESTS 
  test_compare_loop_detection_orb_hfnet.cpp
)
foreach(source_file ${HFNET_TESTS})
  message(STATUS "compiling ${source_file}")
  get_filename_component(target_name ${source_file} NAME_WE)
  add_executable(${target_name} ${source_file})
  target_link_libraries(${target_name} hfnet_features ${CORELIBS})
endforeach()