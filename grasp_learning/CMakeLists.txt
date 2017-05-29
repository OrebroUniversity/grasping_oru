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
  grasp_planner
  hiqp_msgs
  message_generation
)

find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)


add_message_files(
  FILES
  FinishRecording.msg
  StartRecording.msg
  State.msg
)

add_service_files(
  FILES
  QueryNN.srv
  PolicySearch.srv
)



generate_messages(
   DEPENDENCIES
   std_msgs
)


catkin_package(
  CATKIN_DEPENDS roscpp rosbag hiqp_ros controller_manager_msgs gazebo_msgs yumi_hw grasp_planner message_runtime
  INCLUDE_DIRS include
)

include_directories(include
  ${Boost_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
)




## Declare a cpp executable
add_executable(demo_learning
  src/demo_learning.cpp
#  src/look_what_i_found.cpp
)

add_executable(grasp_recorder_node
  src/grasp_recorder_node.cpp src/grasp_recorder.cpp
)

add_executable(run_demo_node
  src/run_demo_node.cpp
)

add_executable(teleop_remapper_node
 src/teleop_remapper.cpp
)


## Specify libraries to link a library or executable target against
target_link_libraries(demo_learning ${catkin_LIBRARIES})

## Specify libraries to link a library or executable target against
target_link_libraries(grasp_recorder_node ${catkin_LIBRARIES})

## Specify libraries to link a library or executable target against
target_link_libraries(run_demo_node ${catkin_LIBRARIES})

target_link_libraries(teleop_remapper_node ${catkin_LIBRARIES} ${Boost_LIBRARIES})
