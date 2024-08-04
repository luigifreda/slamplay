set(HFNET_EXT_INCLUDE_DIRS
  ${fmt_INCLUDE_DIRS}
  #${EIGEN3_INCLUDE_DIR}
  #${Boost_INCLUDE_DIRS}
  #${CHOLMOD_INCLUDE_DIR}
  #${Pangolin_INCLUDE_DIRS}
  #${PCL_INCLUDE_DIRS}
  #${Sophus_INCLUDE_DIRS}
  #${GTSAM_INCLUDE_DIR}
  #${G2O_INCLUDE_DIR}  
  #${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  #${SESYNC_INCLUDE_DIRS}
  #${CSPARSE_INCLUDE_DIR}
  #${DBoW3_INCLUDE_DIRS}
  #${OBINDEX2_INCLUDE_DIR}
  #${IBOW_LCD_INCLUDE_DIR}
  #${OCTOMAP_INCLUDE_DIRS}
  #${GLOG_INCLUDE_DIRS}
  #${GTEST_INCLUDE_DIRS}
  #${GFLAGS_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  ${TensorRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS}
  #${TRACY_INCLUDE_DIR}
  #${Tensorflow_INCLUDE_DIRS}  # may use a different version of Eigen 
  #${PROTOBUF_INCLUDE_DIRS}
)

#file(GLOB_RECURSE SOURCES "./*.cpp")

include_directories(${INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

set(EIGEN_EXTERNAL_INCLUDES "${TENSORFLOW_ROOT}/include/tensorflow/bazel-bin/tensorflow/include/") # to privatly include the tensorflow Eigen

set(HFNET_SOURCES 
${CMAKE_CURRENT_SOURCE_DIR}/extractors/BaseModel.cpp
${CMAKE_CURRENT_SOURCE_DIR}/extractors/HFextractor.cpp
${CMAKE_CURRENT_SOURCE_DIR}/extractors/HFNetTFModel.cpp
${CMAKE_CURRENT_SOURCE_DIR}/extractors/HFNetTFModelV2.cpp
${CMAKE_CURRENT_SOURCE_DIR}/extractors/HFNetRTModel.cpp
#${CMAKE_CURRENT_SOURCE_DIR}/extractors/HFNetVINOModel.cpp
${CMAKE_CURRENT_SOURCE_DIR}/extractors/HFNetSettings.cpp
)

#add_library(hfnet_features ${HFNET_SOURCES})
add_library(hfnet_features STATIC)
target_sources(hfnet_features PRIVATE ${HFNET_SOURCES})
#set_property(TARGET hfnet_features PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${HFNET_EXT_INCLUDE_DIRS} ${EIGEN_EXTERNAL_INCLUDES})
target_include_directories(hfnet_features 
    PUBLIC ${HFNET_EXT_INCLUDE_DIRS} 
    PRIVATE ${EIGEN_EXTERNAL_INCLUDES}
)
target_link_libraries(hfnet_features 
    PUBLIC ${CORELIBS} 
    PRIVATE ${Tensorflow_LIBS}
)

# include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${EXT_LIBS})

if(USE_TENSORFLOW)
    add_executable(test_hfnet_tf_v1 ${CMAKE_CURRENT_SOURCE_DIR}/test_hfnet_tf_v1.cpp)
    target_include_directories(hfnet_features PRIVATE ${EIGEN_EXTERNAL_INCLUDES})    
    target_link_libraries(test_hfnet_tf_v1 hfnet_features PUBLIC ${CORELIBS} PRIVATE ${Tensorflow_LIBS})

    add_executable(test_hfnet_tf_v2 ${CMAKE_CURRENT_SOURCE_DIR}/test_hfnet_tf_v2.cpp)
    target_include_directories(hfnet_features PRIVATE ${EIGEN_EXTERNAL_INCLUDES})
    target_link_libraries(test_hfnet_tf_v2 hfnet_features PUBLIC ${CORELIBS} PRIVATE ${Tensorflow_LIBS})
endif()

if(USE_OPENVINO)
    add_executable(test_hfnet_vino ${CMAKE_CURRENT_SOURCE_DIR}/test_hfnet_vino.cpp)
    target_link_libraries(test_hfnet_vino hfnet_features ${CORELIBS})
endif()

add_executable(test_extractors ${CMAKE_CURRENT_SOURCE_DIR}/test_extractors.cpp)
target_link_libraries(test_extractors hfnet_features ${CORELIBS})

add_executable(test_match_global_feats ${CMAKE_CURRENT_SOURCE_DIR}/test_match_global_feats.cpp)
target_link_libraries(test_match_global_feats hfnet_features ${CORELIBS})

add_executable(test_match_local_feats ${CMAKE_CURRENT_SOURCE_DIR}/test_match_local_feats.cpp)
target_link_libraries(test_match_local_feats hfnet_features ${CORELIBS})
