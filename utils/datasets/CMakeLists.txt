include_directories(${INCLUDE_DIRS} ${EXT_INCLUDE_DIRS})
set(CORELIBS ${LIBS} ${EXT_LIBS})

#file(GLOB_RECURSE SOURCES "./*.cpp")

add_library(bal_dataset BALProblem.cpp)
#target_link_libraries(bal_dataset ${CORELIBS})

add_library(datasets DatasetIo.cpp DatasetIo.h DatasetIoUzh.h DatasetIoKitti.h DatasetIoEuroc.h DatasetIoRosbag.h DatasetIoTumRgbd.h DatasetIoReplica.h)
target_link_libraries(datasets ${CORELIBS})
