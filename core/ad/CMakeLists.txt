include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})


add_subdirectory(common)
add_subdirectory(imu)
add_subdirectory(pointcloud)
add_subdirectory(g2o)
add_subdirectory(io)
add_subdirectory(kdtree)
add_subdirectory(grid)
add_subdirectory(laser_3d)
add_subdirectory(laser_2d)
add_subdirectory(timer)