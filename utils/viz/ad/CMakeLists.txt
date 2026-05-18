add_library(autonomous_driving_viz
        autonomous_driving_viz.cc
        autonomous_driving_viz_impl.cc
        ui_car.cc
        ui_trajectory.cc
        ui_cloud.cc
        )


target_link_libraries(autonomous_driving_viz ${CORELIBS})

# add_executable(ui_test ui/ui_test.cc)
# target_link_libraries(ui_test
#         ${PROJECT_NAME}.tools
#         ${third_party_libs}
#         )
