cmake_minimum_required(VERSION 2.8.3)
project(constraint_map)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

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
find_package(Eigen REQUIRED)
find_package(OpenMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
find_package(Boost REQUIRED)
include_directories(include ${PCL_INCLUDE_DIRS})


## Generate messages in the 'msg' folder
add_message_files(
  FILES
  SimpleOccMapMsg.msg
)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   std_msgs  # Or other packages containing msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS message_runtime std_msgs roscpp tf tf_conversions pcl_ros pcl_conversions
  DEPENDS eigen
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS} include ${Boost_INCLUDE_DIRS}
)
include_directories(include ${EIGEN_INCLUDE_DIRS})
add_definitions(${EIGEN_DEFINITIONS})

## Declare a cpp library
add_library(${PROJECT_NAME}
   src/simple_occ_map.cpp
   src/constraint_map.cpp
   src/region_extraction.cpp
 )

# Declare a cpp executable
add_executable(simple_occ_map_node src/simple_occ_map_node.cpp)
add_executable(compute_constraints src/compute_constraints.cpp)
add_executable(test_regions src/test_regions.cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
add_dependencies(${PROJECT_NAME} constraint_map_generate_messages_cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}
   ${catkin_LIBRARIES} ${Boost_LIBRARIES} 
)

target_link_libraries(simple_occ_map_node
   ${catkin_LIBRARIES} ${PROJECT_NAME}
)
target_link_libraries(compute_constraints
   ${catkin_LIBRARIES} ${PROJECT_NAME}
)
target_link_libraries(test_regions
   ${catkin_LIBRARIES} ${PROJECT_NAME}
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS simple_occ_map simple_occ_map_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_simple_occ_map.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
