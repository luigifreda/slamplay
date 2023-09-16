#set(CMAKE_VERBOSE_MAKEFILE ON)

include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

file(GLOB_RECURSE SOURCES "./*.cpp")

# locally modify the flags to remove march native optimization for gtsam 
# https://bitbucket.org/gtborg/gtsam/issues/414/compiling-with-march-native-results-in
# https://groups.google.com/g/gtsam-users/c/jdySXchYVQg 
string(REPLACE "-march=native" "" MY_LOCAL_C_FLAGS ${MY_C_FLAGS})
string(REPLACE "-march=native" "" MY_LOCAL_CXX_FLAGS ${MY_CXX_FLAGS})
set(CMAKE_C_FLAGS "${MY_LOCAL_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${MY_LOCAL_CXX_FLAGS}")

# Add an executable target for each source file
foreach(source_file ${SOURCES})
  message(STATUS "compiling ${source_file}")
  get_filename_component(target_name ${source_file} NAME_WE)
  add_executable(${target_name} ${source_file})
  target_link_libraries(${target_name} ${CORELIBS})
endforeach()