cmake_minimum_required(VERSION 2.8.3)
project(task_definition_plugins)

set(CMAKE_BUILD_TYPE RelWithDebinfo)
set(CMAKE_CXX_STANDARD 11)
add_definitions(-Wno-missing-braces)

find_package(catkin REQUIRED COMPONENTS
  hiqp_core
  hiqp_ros
  sdf_collision_check
  visualization_msgs
  pluginlib
  tf
  roscpp
)

find_package(Boost REQUIRED COMPONENTS timer)

# find_package(Boost REQUIRED COMPONENTS system)
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES tdef_avoid_collisions_sdf
  CATKIN_DEPENDS hiqp_core sdf_collision_check visualization_msgs pluginlib tf hiqp_ros roscpp
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

add_library(task_definition_plugins src/tdef_avoid_collisions_sdf.cpp src/tdef_meta_task.cpp)
add_dependencies(task_definition_plugins ${catkin_EXPORTED_TARGETS})
target_link_libraries(task_definition_plugins ${catkin_LIBRARIES})

# Some helper programs to add the avoidance task.

add_executable(right_gripper_capsule src/right_gripper_capsule.cpp)
add_dependencies(right_gripper_capsule ${catkin_EXPORTED_TARGETS})
target_link_libraries(right_gripper_capsule ${catkin_LIBRARIES})

add_executable(add_avoidance_cylinders src/add_avoidance_cylinders.cpp)
add_dependencies(add_avoidance_cylinders ${catkin_EXPORTED_TARGETS})
target_link_libraries(add_avoidance_cylinders ${catkin_LIBRARIES})

