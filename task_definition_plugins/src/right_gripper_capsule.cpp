#include <array>
#include <iostream>

#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <hiqp_ros/hiqp_client.h>

int main(int argn, char** args) {
  ros::init(argn, args, "add_avoidance_cylinders");
  std::string res;
  std::cout << "Enter resolution: " << std::endl;
  std::cin >> res;

  ros::NodeHandle nh;
  tf::TransformListener tl;

  hiqp_ros::HiQPClient hiqp_client("yumi", "hiqp_joint_velocity_controller");

  std::vector<std::string> def = {"TDefAvoidCollisionsSDF", res,
                                  "cylinder", "left_gripper"};

  tf::StampedTransform transform_r;

  bool tfAvailable = false;
  while (!tfAvailable && ros::ok()) {
    try {
      tl.waitForTransform("yumi_link_7_l", "yumi_link_tool_l", ros::Time(0),
                          ros::Duration(0.5));
      tl.lookupTransform("yumi_link_7_l", "yumi_link_tool_l", ros::Time(0),
                         transform_r);
    } catch (tf::TransformException& ex) {
      ROS_WARN("TF lookup failed with error: %s", ex.what());
      continue;
    }
    tfAvailable = true;
  }

  tf::Vector3 axis_r = transform_r.getOrigin();
  auto radius = 0.040;

  double length = axis_r.length();
  hiqp_client.setPrimitive("left_gripper", "cylinder", "yumi_link_7_l", true,
                           {1.0, 1.0, 0.0, 0.5},
                           {axis_r.getX(), axis_r.getY(), axis_r.getZ(), 0.00,
                            0.00, 0.00, radius, length+0.05});

  hiqp_client.setTask("avoid_collisions_sdf", 1, true, true, true, def,
                      {"TDynLinear", "5.0"});

  return 0;
}
