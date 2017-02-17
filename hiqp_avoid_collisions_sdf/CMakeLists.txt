cmake_minimum_required(VERSION 2.8.3)
project(hiqp_avoid_collisions_sdf)

set(CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED COMPONENTS
  hiqp_core
  sdf_collision_check
  visualization_msgs
  pluginlib
)

# find_package(Boost REQUIRED COMPONENTS system)
catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES hiqp_avoid_collisions_sdf
  CATKIN_DEPENDS hiqp_core sdf_collision_check visualization_msgs pluginlib
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)


add_library(tdef_avoid_collisions_sdf src/tdef_avoid_collisions_sdf.cpp)
add_dependencies(tdef_avoid_collisions_sdf ${catkin_EXPORTED_TARGETS})
target_link_libraries(tdef_avoid_collisions_sdf ${catkin_LIBRARIES})

