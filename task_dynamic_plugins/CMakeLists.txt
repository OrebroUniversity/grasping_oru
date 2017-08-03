cmake_minimum_required(VERSION 2.8.3)
project(task_dynamic_plugins)

set(CMAKE_BUILD_TYPE RelWithDebinfo)
set(CMAKE_CXX_STANDARD 11)

set(GUROBI_INCLUDE_DIR "$ENV{GUROBI_HOME}/include")
set(GUROBI_LIB_DIR "$ENV{GUROBI_HOME}/lib")
include_directories(${GUROBI_INCLUDE_DIR})
link_directories(${GUROBI_LIB_DIR})

find_package(catkin REQUIRED COMPONENTS
  hiqp_core
  pluginlib
  std_msgs
  grasp_learning
)

# find_package(Boost REQUIRED COMPONENTS system)
catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES hiqp_avoid_collisions_sdf
  CATKIN_DEPENDS hiqp_core pluginlib std_msgs grasp_learning
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)



add_library(task_dynamic_plugins src/tdyn_random.cpp src/tdyn_policy.cpp src/tdyn_linear_policy.cpp src/tdyn_constant.cpp src/tdyn_RBFN.cpp)
add_dependencies(task_dynamic_plugins ${catkin_EXPORTED_TARGETS})
target_link_libraries(task_dynamic_plugins ${catkin_LIBRARIES})
