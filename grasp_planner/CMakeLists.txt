cmake_minimum_required(VERSION 2.8.3)
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE RelWithDebinfo)
project(grasp_planner)

set(CMAKE_CXX_STANDARD 14)
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  eigen_conversions 
  cmake_modules
  constraint_map
  geometry_msgs
  message_generation
  sdf_tracker_grasping
  std_msgs
  hiqp_msgs
  std_srvs
  tf
  pcl_ros
  pcl_conversions
  image_transport
  image_geometry
)

find_package(Boost REQUIRED COMPONENTS thread)
find_package(PCL 1.3 REQUIRED common)

add_service_files(
  FILES
  PlanGrasp.srv
  LoadResource.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs 
  geometry_msgs 
  hiqp_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS constraint_map geometry_msgs message_runtime sdf_tracker_grasping hiqp_msgs std_msgs std_srvs tf eigen_conversions pcl_ros pcl_conversions image_transport image_geometry
  DEPENDS PCL
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)


# Grasp planner library
add_library(grasp_planner src/grasp_planner.cpp)
add_dependencies(grasp_planner grasp_planner_generate_messages_cpp)
target_link_libraries(grasp_planner
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  ${PCL_COMMON_LIBRARIES}
  )

# Grasp planner ROS node.
add_executable(grasp_planner_node src/grasp_planner_node.cc)
add_dependencies(grasp_planner_node grasp_planner_generate_messages_cpp)
target_link_libraries(grasp_planner_node
  grasp_planner
  ${catkin_LIBRARIES}
)

# Tests.
add_executable(run_gplanner_tests src/run_tests.cc)
add_dependencies(run_gplanner_tests grasp_planner_generate_messages_cpp)
target_link_libraries(run_gplanner_tests
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  ${PCL_COMMON_LIBRARIES}
)

# Pointcloud publisher.
add_executable(pointcloud_publisher src/point_cloud_publisher.cpp)
target_link_libraries(pointcloud_publisher
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  ${PCL_COMMON_LIBRARIES}
)
