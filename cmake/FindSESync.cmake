# Locate the g2o libraries
# A general framework for graph optimization.
#
# This module defines
# SESYNC_FOUND, if false, do not try to link against g2o
# SESYNC_LIBRARIES, path to the libg2o
# SESYNC_INCLUDE_DIR, where to find the g2o header files

include(FindPackageHandleStandardArgs)

find_path(SESYNC_LIB_DIR libSESync.so
  ${SESYNC_ROOT}/C++/lib
  ${SESYNC_ROOT}/C++/build/lib
  $ENV{SESYNC_ROOT}/C++/lib
  $ENV{SESYNC_ROOT}/C++/build/lib  
  NO_DEFAULT_PATH
  )
message(STATUS "found SESYNC_LIB_DIR: ${SESYNC_LIB_DIR}")

# Find the header files

find_path(SESYNC_BASE_INCLUDE_DIR SESync/SESync.h
  ${SESYNC_ROOT}/C++/SE-Sync/include
  $ENV{SESYNC_ROOT}/C++/SE-Sync/include
  NO_DEFAULT_PATH
  )
message(STATUS "found SESYNC_BASE_INCLUDE_DIR: ${SESYNC_BASE_INCLUDE_DIR}")  

find_path(SESYNC_OPTIMIZATION_INCLUDE_DIR Optimization/Base/Concepts.h
  ${SESYNC_ROOT}/C++/Optimization/include
  $ENV{SESYNC_ROOT}/C++/Optimization/include
  NO_DEFAULT_PATH
  )
message(STATUS "found SESYNC_OPTIMIZATION_INCLUDE_DIR: ${SESYNC_OPTIMIZATION_INCLUDE_DIR}")  


# Macro to unify finding both the debug and release versions of the
# libraries; this is adapted from the OpenSceneGraph FIND_LIBRARY
# macro.

macro(FIND_SESYNC_LIBRARY MYLIBRARY MYLIBRARYNAME)
  
  find_library(${MYLIBRARY}
    NAMES "${MYLIBRARYNAME}"
    PATHS
    ${SESYNC_ROOT}/C++/build/lib
    ${SESYNC_ROOT}/C++/lib
    $ENV{SESYNC_ROOT}/C++/build/lib
    $ENV{SESYNC_ROOT}/C++/lib
    NO_DEFAULT_PATH
    )

  if(${MYLIBRARY})
    add_library(${MYLIBRARYNAME} UNKNOWN IMPORTED GLOBAL)
    set_target_properties(${MYLIBRARYNAME} PROPERTIES 
      IMPORTED_LOCATION ${${MYLIBRARY}}
      IMPORTED_NO_SONAME ON)
    # https://stackoverflow.com/questions/68164903/cmake-to-link-external-library-with-import-soname-ro-import-location
    #set_property(TARGET ${MYLIBRARYNAME} PROPERTY IMPORTED_NO_SONAME TRUE)
    #message(STATUS "imported location for ${MYLIBRARYNAME}: ${${MYLIBRARY}}")
  endif()
  
endmacro(FIND_SESYNC_LIBRARY LIBRARY LIBRARYNAME)

# Find the core elements
FIND_SESYNC_LIBRARY(SESYNC_LIBRARY SESync)
FIND_SESYNC_LIBRARY(SESYNC_ILDL_LIBRARY ILDL)
FIND_SESYNC_LIBRARY(SESYNC_SCHOL_LIBRARY LSChol)
FIND_SESYNC_LIBRARY(SESYNC_VIZ SESyncViz)

# SESYNC itself declared found if we found the core libraries and at least one solver
set(SESYNC_FOUND "NO")
if(SESYNC_LIBRARY AND SESYNC_ILDL_LIBRARY AND SESYNC_SCHOL_LIBRARY)
  set(SESYNC_FOUND "YES")
endif()


# adaptation for collecting all together the libs 
if(SESYNC_FOUND)
    set(SESYNC_LIBS
        ${SESYNC_LIBRARY}
        ${SESYNC_ILDL_LIBRARY}
        ${SESYNC_SCHOL_LIBRARY}
        ${SESYNC_VIZ}
    )
    set(SESYNC_INCLUDE_DIRS
        ${SESYNC_BASE_INCLUDE_DIR}
        ${SESYNC_OPTIMIZATION_INCLUDE_DIR}
    )
endif()
