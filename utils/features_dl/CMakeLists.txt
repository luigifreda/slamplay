#====================================
# folders to build 

if(WITH_TENSORRT OR WITH_TENSORFLOW)
  add_subdirectory(hfnet)
  add_subdirectory(superpointglue_tensorrt)  
endif()
