#====================================
# folders to build 

# set(FOLDERS_TENSORRT
#   superpointglue_tensorrt
#   hfnet_tensorrt
# )

# foreach(folder ${FOLDERS_TENSORRT})
#   message(STATUS "adding folder: ${folder}")
#   add_subdirectory(${folder})
# endforeach()
    

if(WITH_TENSORRT)
  add_subdirectory(superpointglue_tensorrt)
endif()

if(WITH_TENSORRT)
  add_subdirectory(hfnet)
endif()
