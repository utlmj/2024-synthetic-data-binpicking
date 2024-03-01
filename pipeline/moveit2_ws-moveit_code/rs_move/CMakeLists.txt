cmake_minimum_required(VERSION 3.8)
project(rs_move)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

# Common cmake code applied to all moveit packages
find_package(moveit_common REQUIRED)
moveit_package()

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED system filesystem date_time thread)
find_package(ament_cmake REQUIRED)
find_package(control_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_ros_perception REQUIRED)
find_package(moveit_servo REQUIRED)
#find_package(moveit_task_constructor_core REQUIRED)
find_package(interactive_markers REQUIRED)
find_package(rviz_visual_tools REQUIRED)
find_package(moveit_visual_tools REQUIRED)
find_package(geometric_shapes REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(pluginlib REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
#find_package(csv REQUIRED)

set(THIS_PACKAGE_INCLUDE_DEPENDS
  ament_cmake
  rclcpp
  rclcpp_action
  tf2_geometry_msgs
  tf2_ros
  moveit_core
  #moveit_task_constructor_core
  rviz_visual_tools
  moveit_visual_tools
  moveit_ros_planning_interface
  interactive_markers
  tf2_geometry_msgs
  moveit_ros_planning
  pluginlib
  Eigen3
  Boost
  control_msgs
  moveit_servo
  moveit_msgs
  shape_msgs
  sensor_msgs
  geometry_msgs
)



add_executable(move_group_interface_rs_test  # ORIGINAL HAD INSTEAD OF RS_TEST-- TUTORIAL
    src/move_group_interface_rs_test.cpp) #move_group_interface_tutorial.cpp)
target_include_directories(move_group_interface_rs_test
    PUBLIC include)
ament_target_dependencies(move_group_interface_rs_test
    ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost
    rclcpp shape_msgs)

install(TARGETS move_group_interface_rs_test 
  DESTINATION lib/${PROJECT_NAME}
)


add_executable(rs_move_cam 
    src/rs_move_cam.cpp) 
target_include_directories(rs_move_cam
    PUBLIC include)
ament_target_dependencies(rs_move_cam
    ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost
    rclcpp shape_msgs)

install(TARGETS rs_move_cam
  DESTINATION lib/${PROJECT_NAME}
)


add_executable(param_node src/param_node.cpp)
ament_target_dependencies(param_node ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost)

install(TARGETS
  param_node
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(test_publisher  # ORIGINAL HAD INSTEAD OF RS_TEST-- TUTORIAL
    src/test_publisher.cpp) #move_group_interface_tutorial.cpp)
target_include_directories(test_publisher
    PUBLIC include)
ament_target_dependencies(test_publisher
    ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost)

install(TARGETS test_publisher ### _tutori
  DESTINATION lib/${PROJECT_NAME}
)


add_executable(read_csv  # ORIGINAL HAD INSTEAD OF RS_TEST-- TUTORIAL
    src/read_csv.cpp) #move_group_interface_tutorial.cpp)
target_include_directories(read_csv
    PUBLIC include)
ament_target_dependencies(read_csv
    ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost)

install(TARGETS read_csv ### _tutori
  DESTINATION lib/${PROJECT_NAME}
)


add_executable(read_csv2  # ORIGINAL HAD INSTEAD OF RS_TEST-- TUTORIAL
    src/read_csv2.cpp) #move_group_interface_tutorial.cpp)
target_include_directories(read_csv2
    PUBLIC include)
ament_target_dependencies(read_csv2
    ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost)

install(TARGETS read_csv2 ### _tutori
  DESTINATION lib/${PROJECT_NAME}
)



if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
