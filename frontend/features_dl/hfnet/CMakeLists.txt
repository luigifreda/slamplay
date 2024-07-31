
include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${EXT_LIBS})

#file(GLOB_RECURSE SOURCES "./*.cpp")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(HFNET_SOURCES 
${CMAKE_CURRENT_SOURCE_DIR}/extractors/BaseModel.cpp
${CMAKE_CURRENT_SOURCE_DIR}/extractors/HFextractor.cpp
${CMAKE_CURRENT_SOURCE_DIR}/extractors/HFNetTFModel.cpp
${CMAKE_CURRENT_SOURCE_DIR}/extractors/HFNetTFModelV2.cpp
${CMAKE_CURRENT_SOURCE_DIR}/extractors/HFNetRTModel.cpp
${CMAKE_CURRENT_SOURCE_DIR}/extractors/HFNetVINOModel.cpp
${CMAKE_CURRENT_SOURCE_DIR}/extractors/HFNetSettings.cpp
)

add_library(hfnet_features ${HFNET_SOURCES})
target_link_libraries(hfnet_features ${CORELIBS})



if(${USE_TENSORFLOW})
    add_executable(test_hfnet_tf_v1 ${CMAKE_CURRENT_SOURCE_DIR}/test_hfnet_tf_v1.cpp)
    target_link_libraries(test_hfnet_tf_v1 hfnet_features ${CORELIBS})

    add_executable(test_hfnet_tf_v2 ${CMAKE_CURRENT_SOURCE_DIR}/test_hfnet_tf_v2.cpp)
    target_link_libraries(test_hfnet_tf_v2 hfnet_features ${CORELIBS})
endif()

if(${USE_OPENVINO})
    add_executable(test_hfnet_vino ${CMAKE_CURRENT_SOURCE_DIR}/test_hfnet_vino.cpp)
    target_link_libraries(test_hfnet_vino hfnet_features ${CORELIBS})
endif()

add_executable(test_extractors ${CMAKE_CURRENT_SOURCE_DIR}/test_extractors.cpp)
target_link_libraries(test_extractors hfnet_features ${CORELIBS})

add_executable(test_match_global_feats ${CMAKE_CURRENT_SOURCE_DIR}/test_match_global_feats.cpp)
target_link_libraries(test_match_global_feats hfnet_features ${CORELIBS})

add_executable(test_match_local_feats ${CMAKE_CURRENT_SOURCE_DIR}/test_match_local_feats.cpp)
target_link_libraries(test_match_local_feats hfnet_features ${CORELIBS})
