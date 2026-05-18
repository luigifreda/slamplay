include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

#file(GLOB_RECURSE SOURCES "./*.cpp")

add_library(ad_imu 
    gins_pre_integ.cpp gins_pre_integ.h
    imu_preintegration.cpp imu_preintegration.h
    static_imu_init.cpp static_imu_init.h 
)
target_link_libraries(ad_imu ad_common ${CORELIBS})
