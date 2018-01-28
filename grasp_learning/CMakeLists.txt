cmake_minimum_required(VERSION 2.8.3)
project(grasp_learning)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-Wno-missing-braces")

set(GUROBI_INCLUDE_DIR "$ENV{GUROBI_HOME}/include")
set(GUROBI_LIB_DIR "$ENV{GUROBI_HOME}/lib")
include_directories(${GUROBI_INCLUDE_DIR})
link_directories(${GUROBI_LIB_DIR})

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  hiqp_ros
  gazebo_msgs
  roscpp
  rosbag
  yumi_hw
  hiqp_msgs
  message_generation
)

find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)


add_message_files(
  FILES
  FinishRecording.msg
  StartRecording.msg
  AddNoise.msg
  State.msg
  RobotState.msg
)

add_service_files(
  FILES
  QueryNN.srv
  CallRBFN.srv
  SetRBFN.srv
  PolicySearch.srv
  GetNetworkWeights.srv
)

generate_messages(
   DEPENDENCIES
   std_msgs
)


catkin_package(
  CATKIN_DEPENDS roscpp rosbag hiqp_ros controller_manager_msgs gazebo_msgs yumi_hw message_runtime
  INCLUDE_DIRS include
)

include_directories(include
  ${Boost_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
)




add_executable(run_demo_node
  src/run_demo_node.cpp
)

add_executable(demo_learn_manifold
  src/demo_learn_manifold.cpp src/fileHandler.cpp
)

add_executable(RBFNetwork
  src/RBFNetwork.cpp src/power.cpp src/MultiVariateGaussian.cpp src/fileHandler.cpp
)


## Specify libraries to link a library or executable target against
target_link_libraries(run_demo_node ${catkin_LIBRARIES})

## Specify libraries to link a library or executable target against
target_link_libraries(demo_learn_manifold ${catkin_LIBRARIES})

## Specify libraries to link a library or executable target against
target_link_libraries(RBFNetwork ${catkin_LIBRARIES})
