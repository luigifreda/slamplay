include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

#file(GLOB_RECURSE SOURCES "./*.cpp")

add_library(ad_g2o 
        g2o_types.cpp g2o_types.h
        g2o_imu_types.cpp g2o_imu_types.h)
target_link_libraries(ad_g2o ad_common ${CORELIBS})
