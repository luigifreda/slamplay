
#set(CMAKE_VERBOSE_MAKEFILE ON)

# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling 
# NOTE: These lines do not work!
#set(CMAKE_SKIP_BUILD_RPATH OFF)
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
#list(APPEND CMAKE_BUILD_RPATH ${G2O_LIB_DIR})
#list(APPEND CMAKE_INSTALL_RPATH ${G2O_LIB_DIR}) 

include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

#message(STATUS "linking against ${CORELIBS}")

file(GLOB_RECURSE SOURCES "./*.cpp")

# Add an executable target for each source file
foreach(source_file ${SOURCES})
  message(STATUS "compiling ${source_file}")
  get_filename_component(target_name ${source_file} NAME_WE)
  add_executable(${target_name} ${source_file})
  target_link_libraries(${target_name} ${CORELIBS}) 
endforeach()