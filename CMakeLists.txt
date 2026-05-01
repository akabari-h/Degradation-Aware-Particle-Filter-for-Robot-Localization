cmake_minimum_required(VERSION 3.8)
project(rsn_p)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  install(DIRECTORY config launch meshes urdf worlds models scripts
  DESTINATION share/${PROJECT_NAME}/
)

install(PROGRAMS
  scripts/camera_degradation_node.py
  scripts/ultrasonic_degradation_node.py
  scripts/confidence_monitor_node.py
  scripts/particle_filter_node.py
  scripts/lidar_degradation_node.py
  scripts/odom_tf_publisher.py
  
  scripts/plot_particles.py
  DESTINATION lib/${PROJECT_NAME}
)

  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
