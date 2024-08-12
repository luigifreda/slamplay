#====================================
# folders to build 

if(WITH_TENSORRT OR WITH_TENSORFLOW)
  add_subdirectory(hfnet)
endif()
