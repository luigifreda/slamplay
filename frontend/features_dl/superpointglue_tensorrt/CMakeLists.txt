include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

add_executable(test_superpointglue_image test_superpointglue_image.cpp)
target_link_libraries(test_superpointglue_image superpointglue ${CORELIBS})

add_executable(test_superpointglue_sequence test_superpointglue_sequence.cpp)
target_link_libraries(test_superpointglue_sequence  superpointglue ${CORELIBS})
