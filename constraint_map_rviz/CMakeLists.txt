cmake_minimum_required(VERSION 2.8.3)
project(constraint_map_rviz)

set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS rviz constraint_map)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)

# add_definitions(-DQT_NO_KEYWORDS)

# This must be corrected.
catkin_package()

qt5_wrap_cpp(MOC_FILES
  src/occ_display.hpp
)

## Here we specify the list of source files, including the output of
## the previous command which is stored in ``${MOC_FILES}``.
set(SOURCE_FILES
  src/occ_display.cpp
  src/occ_visual.cpp
  ${MOC_FILES}
)

include_directories(
  ${Qt5Core_INCLUDE_DIRS}
  ${Qt5Gui_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${catkin_INCLUDE_DIRS}
)

## An rviz plugin is just a shared library, so here we declare the
## library to be called ``${PROJECT_NAME}`` (which is
## "rviz_plugin_tutorials", or whatever your version of this project
## is called) and specify the list of source files we collected above
## in ``${SOURCE_FILES}``.
add_library(${PROJECT_NAME} ${SOURCE_FILES})

## Link the library with whatever Qt libraries have been defined by
## the ``find_package(Qt4 ...)`` line above, and with whatever libraries
## catkin has included.
##
## Although this puts "rviz_plugin_tutorials" (or whatever you have
## called the project) as the name of the library, cmake knows it is a
## library and names the actual file something like
## "librviz_plugin_tutorials.so", or whatever is appropriate for your
## particular OS.
target_link_libraries(${PROJECT_NAME}
  Qt5::Core
  Qt5::Gui
  ${catkin_LIBRARIES})

## Install rules
install(TARGETS
  ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES 
  plugin_description.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

#install(DIRECTORY media/
#  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/media)

#install(DIRECTORY icons/
#  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/icons)

#install(PROGRAMS scripts/send_test_msgs.py
#  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

