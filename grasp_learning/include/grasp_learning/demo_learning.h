#ifndef DEMO_LEARNING_H
#define DEMO_LEARNING_H

#include <ros/ros.h>
#include <rosbag/bag.h>

#include <controller_manager_msgs/SwitchController.h>
#include <hiqp_ros/hiqp_client.h>
#include <sensor_msgs/JointState.h>
#include <std_srvs/Empty.h>
#include <tf2_msgs/TFMessage.h>
#include <std_msgs/Empty.h>
#include <grasp_learning/StartRecording.h>
#include <grasp_learning/FinishRecording.h>
#include <grasp_learning/PolicySearch.h>
#include <grasp_learning/State.h>

#include <std_msgs/Float64MultiArray.h>

#include <Eigen/Core>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <vector>
#include <cmath>        // std::abs
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <functional>

namespace demo_learning {

#define DYNAMICS_GAIN 1.5

  struct GraspInterval {

    hiqp_msgs::Primitive plane;

  std::string obj_frame_;  // object frame
  std::string e_frame_;    // endeffector frame
  Eigen::Vector3d e_;      // endeffector point expressed in e_frame_
  float angle;
  bool isSphereGrasp, isDefaultGrasp;
};


class DemoLearning {
public:
  DemoLearning();


  template <typename ROSMessageType>
  int addSubscription(ros::NodeHandle& controller_nh, const std::string& topic_name, unsigned int buffer_size){
    ros::Subscriber sub;
    sub = controller_nh.subscribe(topic_name, buffer_size,  &DemoLearning::topicCallback<ROSMessageType>, this);
    ROS_INFO_STREAM("Subsribed to topic '" << topic_name << "'");
    subs_.push_back(sub);
  }


  template <typename ROSMessageType>
  void topicCallback(const ROSMessageType& msg);

private:

  std::default_random_engine generator;
  std::normal_distribution<double> dist;


  std::vector<ros::Subscriber> subs_;

  std::vector< sensor_msgs::JointState > joint_state_vec_;
  std::vector< hiqp_msgs::TaskMeasures > task_dynamics_vec_;
  std::vector< std_msgs::Float64MultiArray > joint_effort_vec_;
  std::vector<double> action_vec_;
  unsigned int n_jnts;
  std::vector<std::string> link_frame_names;

  ros::NodeHandle nh_;
  ros::NodeHandle n_;


  ros::ServiceClient client_Policy_Search_;


  hiqp_ros::HiQPClient hiqp_client_;

  bool with_gazebo_;  ///<indicate whether the node is run in simulation
  bool generalize_policy_; // Indicate wheter we want to check how well our learned policy generalizes
  bool record_ = false;
  bool converged_policy_ = false;
  std::string input_data_file_;
  std::string output_data_file_;

  double reward_treshold_ = 8;
  std::vector<double> reward_vec_;
  unsigned int num_record_=0;
  // object
  Eigen::VectorXd t_prog_prev_;

  GraspInterval grasp_horizontal_;
  GraspInterval grasp_vertical_;

  /// Clients to other nodes
  ros::ServiceClient set_gazebo_physics_clt_;

  /// Servers
  ros::ServiceServer start_demo_srv_;
  //Publisher oublishing an empty message when the demo grasp is over
  ros::Publisher start_recording_;
  ros::Publisher finish_recording_;
  ros::Publisher run_new_episode_;

  ros::Publisher sample_And_Reweight_;

  grasp_learning::StartRecording start_msg_;
  grasp_learning::FinishRecording finish_msg_;
  //Empty message published by finished_grasping_
  std_msgs::Empty empty_msg_;
  //** Manipulator joint configuration while moving the forklift */
  std::vector<double> transfer_config_;
  //** Manipulator joint configuration prior to reach-to-grasp */
  std::vector<double> sensing_config_;

  //**First deactivates the HQP control scheme (the controller will output zero
  // velocity commands afterwards) and then calls a ros::shutdown */
  void safeShutdown();
  //**like shutdown but we can run again */
  void safeReset();

  bool doGraspAndLift();

  bool startDemo(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);

  void updatePolicy();

  std::vector<double> generateStartPosition(std::vector<double> vec);

  std::vector<double> getActions();

  void printVector(const std::vector<double>& vec);

  std::vector<grasp_learning::State> getStates();

};

}  // end namespace hqp controllers

#endif
