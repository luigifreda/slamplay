
# include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
include_directories(${INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

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

# 

add_executable(test_and_compare_extractors ${CMAKE_CURRENT_SOURCE_DIR}/test_and_compare_extractors.cpp)
target_include_directories(test_and_compare_extractors PRIVATE ${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
target_link_libraries(test_and_compare_extractors hfnet_features ${CORELIBS})

add_executable(test_and_compare_matchers ${CMAKE_CURRENT_SOURCE_DIR}/test_and_compare_matchers.cpp)
target_include_directories(test_and_compare_matchers PRIVATE ${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
target_link_libraries(test_and_compare_matchers hfnet_features ${CORELIBS})