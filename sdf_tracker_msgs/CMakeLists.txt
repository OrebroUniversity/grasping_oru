cmake_minimum_required(VERSION 2.8.3)
project(sdf_tracker_msgs)

find_package(catkin REQUIRED
  message_generation
  std_msgs
  geometry_msgs
)

add_message_files(
  FILES
  SDFMap.msg
)

add_service_files(
  FILES
  GetSDFMap.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)
