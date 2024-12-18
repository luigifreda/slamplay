cmake_minimum_required(VERSION 2.8.6)
project(console_bridge)

set (CONSOLE_BRIDGE_MAJOR_VERSION 0)
set (CONSOLE_BRIDGE_MINOR_VERSION 4)
set (CONSOLE_BRIDGE_PATCH_VERSION 3)

set (CONSOLE_BRIDGE_VERSION ${CONSOLE_BRIDGE_MAJOR_VERSION}.${CONSOLE_BRIDGE_MINOR_VERSION}.${CONSOLE_BRIDGE_PATCH_VERSION})
message (STATUS "${PROJECT_NAME} version ${CONSOLE_BRIDGE_VERSION}")

include(GNUInstallDirs)
include(GenerateExportHeader)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW)
endif()

# If compiler support symbol visibility, enable it.
include(CheckCCompilerFlag)
check_c_compiler_flag(-fvisibility=hidden HAS_VISIBILITY)
if (HAS_VISIBILITY)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
endif()

if(NOT WIN32)
  # Use c++11 compiler flag, since it's needed for console.cpp
  # It isn't in a header file, so it doesn't need to be exported
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CMAKE_CXX_FLAGS}")

if(MSVC OR MSVC90 OR MSVC10)
  set(MSVC ON)
endif (MSVC OR MSVC90 OR MSVC10)

include_directories(include ${PROJECT_BINARY_DIR})

if(NOT DEFINED BUILD_SHARED_LIBS)
  option(BUILD_SHARED_LIBS "Build dynamically-linked binaries" ON)
endif()

# Control where libraries and executables are placed during the build
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

add_library(${PROJECT_NAME} src/console.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION
               ${CONSOLE_BRIDGE_MAJOR_VERSION}.${CONSOLE_BRIDGE_MINOR_VERSION})
generate_export_header(${PROJECT_NAME}
    EXPORT_MACRO_NAME CONSOLE_BRIDGE_DLLAPI)

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}-targets 
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h")

install(FILES ${PROJECT_BINARY_DIR}/${PROJECT_NAME}_export.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(WIN32 AND NOT CYGWIN)
  set(CMAKE_CONFIG_INSTALL_DIR CMake)
else()
  set(CMAKE_CONFIG_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}/cmake/)
endif()

include(CMakePackageConfigHelpers)
set(PKG_NAME ${PROJECT_NAME})
set(PKG_LIBRARIES ${PROJECT_NAME})
set(cmake_conf_file "${PROJECT_NAME}-config.cmake")
configure_package_config_file("${cmake_conf_file}.in" "${CMAKE_BINARY_DIR}/${cmake_conf_file}"
                              INSTALL_DESTINATION ${CMAKE_CONFIG_INSTALL_DIR}
                              PATH_VARS CMAKE_INSTALL_FULL_INCLUDEDIR
                              NO_SET_AND_CHECK_MACRO
                              NO_CHECK_REQUIRED_COMPONENTS_MACRO)
set(cmake_conf_version_file "${PROJECT_NAME}-config-version.cmake")
# Use write_basic_package_version_file to generate a ConfigVersion file that
# allow users of gazebo to specify the API or version to depend on
# TODO: keep this instruction until deprecate Ubuntu/Precise and update with
# https://github.com/Kitware/CMake/blob/v2.8.8/Modules/CMakePackageConfigHelpers.cmake
include(WriteBasicConfigVersionFile)
write_basic_config_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${cmake_conf_version_file}
  VERSION "${CONSOLE_BRIDGE_VERSION}"
  COMPATIBILITY SameMajorVersion)

install(EXPORT ${PROJECT_NAME}-targets
  FILE ${PROJECT_NAME}-targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_CONFIG_INSTALL_DIR})
install(FILES
        "${CMAKE_BINARY_DIR}/${cmake_conf_file}"
        "${CMAKE_BINARY_DIR}/${cmake_conf_version_file}"
  DESTINATION ${CMAKE_CONFIG_INSTALL_DIR} COMPONENT cmake)

if (NOT MSVC)
  set(PKG_DESC "Console Bridge")
  set(PKG_CB_LIBS "-l${PROJECT_NAME}")
  set(pkg_conf_file "console_bridge.pc")
  configure_file("${pkg_conf_file}.in" "${CMAKE_BINARY_DIR}/${pkg_conf_file}" @ONLY)
  install(FILES "${CMAKE_BINARY_DIR}/${pkg_conf_file}"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig/ COMPONENT pkgconfig)
endif()

SET_DIRECTORY_PROPERTIES(PROPERTIES
  ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/console_bridge-config.cmake
  ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/console_bridge.pc)

enable_testing()
add_subdirectory(test)
