
include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

enable_testing()

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(apps)

get_current_dir_name(CURRENT_DIR_NAME) # macro from cmake/macros.cmake
# in order to keep the same relative file paths:
# 1) add a symbolic link to the config file in the binary folder (build folder where we build everything)
file(CREATE_LINK ${CMAKE_CURRENT_LIST_DIR}/config ${CMAKE_BINARY_DIR}/${CURRENT_DIR_NAME}/config SYMBOLIC)
# 2) add a symbolic link to the dataset files in the binary folder (build folder where we build everything)
file(CREATE_LINK ${CMAKE_CURRENT_LIST_DIR}/dataset ${CMAKE_BINARY_DIR}/${CURRENT_DIR_NAME}/dataset SYMBOLIC)