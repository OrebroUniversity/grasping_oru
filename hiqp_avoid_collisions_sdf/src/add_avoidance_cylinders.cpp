#include <array>

#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <hiqp_ros/hiqp_client.h>

int main(int argn, char** args) {

  ros::init(argn, args, "add_avoidance_cylinders");
  ros::NodeHandle nh;
  tf::TransformListener tl;
  
  hiqp_ros::HiQPClient hiqp_client("yumi", "hiqp_joint_velocity_controller");

  std::array<std::string, 3> rh_frames = {"yumi_link_6_r", "yumi_link_5_r", "yumi_link_4_r"};
  std::array<std::string, 3> lh_frames = {"yumi_link_6_l", "yumi_link_5_l", "yumi_link_4_l"};

  for(auto frame : rh_frames) {
    
  }
    
}
