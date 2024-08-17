
include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
#set(CORELIBS ${LIBS} ${EXT_LIBS})
set(CORELIBS  ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES} ${TORCH_LIBRARIES} ${OpenCV_LIBS})

if(WITH_TENSORRT AND WITH_TORCH)
  set(EXECUTABLES
  test_sam_auto_segment
  test_sam_sample_with_mouse
  test_sam_video_segment
  )

  foreach(EXECUTABLE ${EXECUTABLES})
    add_executable(${EXECUTABLE} ${EXECUTABLE}.cpp)
    #target_include_directories(${EXECUTABLE} PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(${EXECUTABLE} PRIVATE ${CORELIBS})
    #set_property(TARGET ${EXECUTABLE} PROPERTY CXX_STANDARD 17)
  endforeach()  
endif()  

