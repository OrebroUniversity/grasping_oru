cmake_minimum_required(VERSION 2.8.3)
project(obstacle_avoidance_gazebo)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  gazebo_ros
  gazebo_msgs
  grasp_planner
  hiqp_demos
)

find_package(gazebo REQUIRED)
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

catkin_package(
  LIBRARIES contact
  CATKIN_DEPENDS roscpp gazebo_msgs gazebo_ros grasp_planner hiqp_demos
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

link_directories(${GAZEBO_LIBRARY_DIRS})

add_library(contact src/contact_plugin.cc)
target_link_libraries(contact ${GAZEBO_libraries})
