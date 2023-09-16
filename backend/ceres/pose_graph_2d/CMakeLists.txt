add_executable(pose_graph_2d
  angle_manifold.h
  normalize_angle.h
  pose_graph_2d.cpp
  pose_graph_2d_error_term.h
  types.h)
target_link_libraries(pose_graph_2d PRIVATE Ceres::ceres gflags)

# copy python script into binary folder 
add_custom_target(copy_python_script ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/plot_results.py)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/plot_results.py
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/plot_results.py
                                                    ${CMAKE_CURRENT_BINARY_DIR}/plot_results.py
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/plot_results.py)