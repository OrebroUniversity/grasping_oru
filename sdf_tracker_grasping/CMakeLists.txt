cmake_minimum_required(VERSION 2.8.3)
#set(CMAKE_BUILD_TYPE Release) 
set(CMAKE_BUILD_TYPE RelWithDebinfo) 
project(sdf_tracker_grasping)

find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS roscpp visualization_msgs std_srvs std_msgs cv_bridge constraint_map sdf_tracker_msgs cmake_modules eigen_conversions)
find_package(VTK REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

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

catkin_package(
    DEPENDS Eigen3 VTK OpenCV
    CATKIN_DEPENDS roscpp visualization_msgs std_srvs std_msgs cv_bridge constraint_map sdf_tracker_msgs eigen_conversions
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
)


include_directories(
  include
  ${Eigen3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_definitions(${EIGEN_DEFINITIONS})

add_library(${PROJECT_NAME} src/sdf_tracker.cpp)
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${catkin_LIBRARIES} ${VTK_LIBRARIES})

add_executable(sdf_tracker_grasping_node src/sdf_tracker_node.cpp)
target_link_libraries(sdf_tracker_grasping_node ${Boost_LIBRARIES} ${catkin_LIBRARIES} ${PROJECT_NAME})

# TODO: fill in what other packages will need to use this package
install(TARGETS sdf_tracker_grasping_node ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
       )

install(DIRECTORY include/sdf_tracker/
	DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
       )
install(FILES
   launch/launch_sdf_tracker.launch
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

