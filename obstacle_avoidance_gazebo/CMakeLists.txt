cmake_minimum_required(VERSION 2.8.3)
project(obstacle_avoidance_gazebo)

find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  grasp_planner
  hiqp_demos
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
