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

  std::array<std::string, 4> rh_frames = {"yumi_link_4_r", "yumi_link_5_r",
                                          "yumi_link_6_r", "yumi_link_7_r"};
  std::array<std::string, 4> lh_frames = {"yumi_link_4_l", "yumi_link_5_l",
                                          "yumi_link_6_l", "yumi_link_7_l"};

  // std::array<std::string, 2> rh_frames = {"yumi_link_7_r", "yumi_link_6_r"};
  // std::array<std::string, 2> lh_frames = {"yumi_link_7_l", "yumi_link_6_l"};

  std::vector<std::string> def = {"TDefAvoidCollisionsSDF"};

  for (size_t i = 0; i < rh_frames.size() - 1; i++) {
    tf::StampedTransform transform_r, transform_l;

    bool tfAvailable = false;
    while (!tfAvailable) {
      try {
        tl.waitForTransform(rh_frames[i], rh_frames[i + 1], ros::Time(0),
                            ros::Duration(0.5));
        tl.lookupTransform(rh_frames[i], rh_frames[i + 1], ros::Time(0),
                           transform_r);
      } catch (tf::TransformException& ex) {
        ROS_WARN("TF lookup failed with error: %s", ex.what());
        continue;
      }
      tfAvailable = true;
    }

    tfAvailable = false;
    while (!tfAvailable) {
      try {
        tl.waitForTransform(lh_frames[i], lh_frames[i + 1], ros::Time(0),
                            ros::Duration(0.5));
        tl.lookupTransform(lh_frames[i], lh_frames[i + 1], ros::Time(0),
                           transform_l);

      } catch (tf::TransformException& ex) {
        ROS_WARN("TF lookup failed with error: %s", ex.what());
        continue;
      }
      tfAvailable = true;
    }

    tf::Vector3 axis_r = transform_r.getOrigin();
    tf::Vector3 axis_l = transform_l.getOrigin();

    double length = axis_r.length();
    hiqp_client.setPrimitive(rh_frames[i], "cylinder", rh_frames[i], true,
                             {1.0, 1.0, 0.0, 0.5},
                             {axis_r.getX(), axis_r.getY(), axis_r.getZ(), 0.00,
                              0.00, 0.00, 0.06, length});

    length = axis_l.length();
    hiqp_client.setPrimitive(lh_frames[i], "cylinder", lh_frames[i], true,
                             {1.0, 1.0, 0.0, 0.5},
                             {axis_l.getX(), axis_l.getY(), axis_l.getZ(), 0.00,
                              0.00, 0.00, 0.06, length});

    def.push_back("cylinder");
    def.push_back(rh_frames[i]);
    def.push_back("cylinder");
    def.push_back(lh_frames[i]);
  }

  hiqp_client.setTask("avoid_collisions_sdf", 1, true, true, true, def,
                      {"TDynLinear", "5.0"});

  return 0;
}
