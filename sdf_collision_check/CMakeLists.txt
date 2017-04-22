cmake_minimum_required(VERSION 2.8.3)
project(sdf_collision_check)

set(CMAKE_BUILD_TYPE RelWithDebinfo)
set(CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  sdf_tracker_msgs
  tf
  tf_conversions
  eigen_conversions
  std_srvs
  message_generation
)

add_message_files(FILES
  SDFGradients.msg
)

generate_messages(DEPENDENCIES geometry_msgs)

find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES sdf_collision_checker
  CATKIN_DEPENDS roscpp sdf_tracker_msgs tf tf_conversions std_srvs eigen_conversions message_runtime geometry_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_library(sdf_collision_checker src/sdf_collision_checker.cpp)
add_dependencies(sdf_collision_checker ${catkin_EXPORTED_TARGETS})
