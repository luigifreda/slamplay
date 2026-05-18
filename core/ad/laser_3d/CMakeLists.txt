include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

#file(GLOB_RECURSE SOURCES "./*.cpp")

add_library(ad_laser_3d 
        icp_3d.cpp
        ndt_3d.cpp
        gen_simu_data.cpp
        ndt_lo_direct.cpp
        ndt_lo_incremental.cpp
        ndt_inc.cpp
        loam_like/feature_extraction.cpp
        loam_like/loam_like_odom.cpp

        lio_loosely_coupled/lio_loosely_coupled.cpp
        
        message_sync/message_sync.cpp

        lio_iekf/lio_iekf.cpp
        lio_preinteg/lio_preinteg.cpp
        )
target_link_libraries(ad_laser_3d ad_common ad_pointcloud ad_g2o ad_imu ad_timer ${CORELIBS})
