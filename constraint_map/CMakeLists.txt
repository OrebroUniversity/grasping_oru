cmake_minimum_required(VERSION 2.8.3)
project(constraint_map)
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  message_generation
  std_msgs
  roscpp
  tf
  tf_conversions
  pcl_ros
  pcl_conversions
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
find_package(Boost REQUIRED)
include_directories(include ${PCL_INCLUDE_DIRS})

if(NOT Eigen3_INCLUDE_DIRS)              
    if(NOT EIGEN3_INCLUDE_DIR)               
		message(FATAL_ERROR "Undefined EIGEN include dir env var.")
	else()
		set(Eigen3_INCLUDE_DIRS "${EIGEN3_INCLUDE_DIR}")
		message(STATUS "EIGEN3 include dir found")
	endif()
else()
    message(STATUS "EIGEN3 include dir found")
endif()

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  SimpleOccMapMsg.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   std_msgs  # Or other packages containing msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS message_runtime std_msgs roscpp tf tf_conversions pcl_ros pcl_conversions
  DEPENDS Eigen3 Boost
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
)

add_definitions(${EIGEN_DEFINITIONS})

## Declare a cpp library
add_library(${PROJECT_NAME}
  src/simple_occ_map.cpp
  src/constraint_map.cpp
  src/region_extract_distance_field.cpp
  src/region_extract_segment_tree.cpp
)

add_dependencies(${PROJECT_NAME} constraint_map_generate_messages_cpp)

target_link_libraries(${PROJECT_NAME}
   ${catkin_LIBRARIES} ${Boost_LIBRARIES} 
)


# Declare a cpp executable
add_executable(simple_occ_map_node src/simple_occ_map_node.cpp)
add_executable(compute_constraints src/compute_constraints.cpp)
add_executable(test_regions src/test_regions.cpp)

target_link_libraries(simple_occ_map_node
   ${catkin_LIBRARIES} ${PROJECT_NAME}
)
target_link_libraries(compute_constraints
   ${catkin_LIBRARIES} ${PROJECT_NAME}
)
target_link_libraries(test_regions
   ${catkin_LIBRARIES} ${PROJECT_NAME}
)
