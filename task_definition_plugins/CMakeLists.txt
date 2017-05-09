cmake_minimum_required(VERSION 2.8.3)
project(task_definition_plugins)

set(CMAKE_BUILD_TYPE RelWithDebinfo)
set(CMAKE_CXX_STANDARD 11)
    
set(GUROBI_INCLUDE_DIR "$ENV{GUROBI_HOME}/include")
set(GUROBI_LIB_DIR "$ENV{GUROBI_HOME}/lib")
include_directories(${GUROBI_INCLUDE_DIR})
link_directories(${GUROBI_LIB_DIR})

find_package(catkin REQUIRED COMPONENTS
  hiqp_core
  sdf_collision_check
  visualization_msgs
  pluginlib
  std_msgs
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


add_library(task_definition_plugins src/tdef_avoid_collisions_sdf.cpp src/tdef_meta_task.cpp)
add_dependencies(task_definition_plugins ${catkin_EXPORTED_TARGETS})
target_link_libraries(task_definition_plugins ${catkin_LIBRARIES})

