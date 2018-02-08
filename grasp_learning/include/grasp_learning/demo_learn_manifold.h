#pragma once

#include <ros/ros.h>
#include <rosbag/bag.h>

#include <controller_manager_msgs/SwitchController.h>
#include <sensor_msgs/JointState.h>
#include <std_srvs/Empty.h>
#include <tf2_msgs/TFMessage.h>
#include <std_msgs/Empty.h>
#include <std_msgs/String.h>
#include <std_msgs/Float64MultiArray.h>
#include <control_msgs/JointControllerState.h>
#include <gazebo_msgs/SetPhysicsProperties.h>

#include <hiqp_ros/hiqp_client.h>
#include <yumi_hw/YumiGrasp.h>

#include <grasp_learning/StartRecording.h>
#include <grasp_learning/FinishRecording.h>
#include <grasp_learning/RobotState.h>
#include <grasp_learning/PolicySearch.h>
#include <grasp_learning/AddNoise.h>
#include <grasp_learning/PolicySearch.h>
#include <grasp_learning/SetRBFN.h>
#include <grasp_learning/GetNetworkWeights.h>
#include <grasp_learning/fileHandler.h>


#include <Eigen/Core>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/assign/std/vector.hpp>
#include <vector>
#include <cmath>        // std::abs
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <functional>
#include <math.h>
#include <time.h>
#include <limits>
#include <numeric>
#include <iterator>

namespace demo_learning {

#define GRASP_THRESHOLD 0.005
#define DYNAMICS_GAIN 1.5

struct GraspInterval {

  hiqp_msgs::Primitive plane;

  std::string obj_frame_;  // object frame
  std::string e_frame_;    // endeffector frame
  Eigen::Vector3d e_;      // endeffector point expressed in e_frame_
  float angle;
  bool isSphereGrasp, isDefaultGrasp;
};


class DemoLearnManifold {
 public:
  DemoLearnManifold();


  template <typename ROSMessageType>
  int addSubscription(ros::NodeHandle& controller_nh, const std::string& topic_name, unsigned int buffer_size) {
    ros::Subscriber sub;
    sub = controller_nh.subscribe(topic_name, buffer_size,  &DemoLearnManifold::topicCallback<ROSMessageType>, this);
    ROS_INFO_STREAM("Subsribed to topic '" << topic_name << "'");
    subs_.push_back(sub);
  }


  template <typename ROSMessageType>
  void topicCallback(const ROSMessageType& msg);

  void robotStateCallback(const grasp_learning::RobotState::ConstPtr& msg);

  void robotCollisionCallback(const std_msgs::Empty::ConstPtr& msg);

  void graspStateCallback(const sensor_msgs::JointState::ConstPtr& msg);

  //**First deactivates the HQP control scheme (the controller will output zero
  // velocity commands afterwards) and then calls a ros::shutdown */
  void safeShutdown();
  //**like shutdown but we can run again */
  void safeReset();


  bool doGraspAndLiftNullspace();

  bool doGraspAndLiftTaskspace();

  void resetTrial();

  bool pictureMode(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);

  bool startDemo(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);

  bool runDemo(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);

  bool pauseDemo(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);

  bool setPolicyConverged(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);

  void updatePolicy();

  std::vector<double> generateStartPosition(std::vector<double> vec);

  void printVector(const std::vector<double>& vec);

  double sumVector(const std::vector<double>& vec);

  void addNoise();

  void setRBFNetwork();

  void calculateReward();

  void calculateReachingReward();

  bool successfulGrasp();

  bool isCollision();

  std::vector<double> normalizeVector(const std::vector<double>& v);

  double pointToPointDist(std::vector<double> point1, std::vector<double> point2);

  double pointToLineDist(std::vector<double> point, std::vector<double> line);

  double pointToPlaneDist(std::vector<double> point, std::vector<double> plane);

  double vectorLength(const std::vector<double>& vec);

  void resetMatrix(std::vector<std::vector<double>>& matrix);

  std::vector<double> calcJointTraj();

  std::vector<double> calcJointVel();

  double calcJointMovementOneTimeStep(unsigned int i);

  double calcJointVelocityOneTimeStep(unsigned int i);

  template<typename T>
  Eigen::Map<Eigen::VectorXd> convertToEigenVector(std::vector<T> vec);

  template<typename T>
  Eigen::MatrixXd convertToEigenMatrix(std::vector<std::vector<T>> data);

  void visualizeKernels();

  bool createFiles(std::string);

  void saveTrialDataToFile();

  template<typename T>
  void saveDataToFile(std::string filename, T, bool);

  template<typename T>
  std::vector<T> accumulateVector(std::vector<T> vec);


 private:

  fileHandler fileHandler_;

  std::string relativePath;

  std::string task_;

  std::string rewardFile;
  std::string finalRewardFile;
  std::string normalizedRewardFile;

  std::string pointToLineDistFile;
  std::string pointToPointDistFile;

  std::string jointVelFile;
  std::string jointVelMessageFile;
  std::string jointVelAccFile;

  std::string jointTrajLengthFile;
  std::string jointTrajAccFile;

  std::string gripperPosFile;

  std::string samplingTimeFile;

  std::string trialDataFile;

  std::string graspSuccessFile;
  std::default_random_engine generator;
  std::normal_distribution<double> dist;


  std::vector<ros::Subscriber> subs_;

  unsigned int n_jnts;


  std::vector<double> PCofObject;
  ros::NodeHandle nh_;
  ros::NodeHandle n_;

  hiqp_ros::HiQPClient hiqp_client_;

  bool with_gazebo_;  ///<indicate whether the node is run in simulation
  double decay_rate_;
  double exec_time_;
  double manifold_height_;
  double manifold_radius_;
  double object_radius_;
  std::vector<double> manifoldPos;


  bool demo_running_ = false;
  
  int numTrial_ = 1;
  int maxNumTrials_ = 0;

  int numRollouts_ = 1;
  int maxRolloutsPerTrial_ = 0;

  bool policyConverged_ = false;

  bool collision_ = false;

  bool nullspace_ = true;

  bool init = true;

  bool run_demo_ = true;

  double unsuccessful_grasp_ = 0;
  // object

  int graspFail = 1;

  GraspInterval grasp_;

  /// Clients to other nodes
  ros::ServiceClient set_gazebo_physics_clt_;
  ros::ServiceClient policy_search_clt_;
  ros::ServiceClient add_noise_clt_;
  ros::ServiceClient set_RBFN_clt_;
  ros::ServiceClient get_network_weights_clt_;
  ros::ServiceClient vis_kernel_mean_clt_;
  ros::ServiceClient reset_RBFN_clt_;
  ros::ServiceClient start_demo_clt_;


  ros::Subscriber gripper_pos;
  ros::Subscriber robot_collision;
  ros::Subscriber gripper_state;

  ros::ServiceClient close_gripper_clt_;
  ros::ServiceClient open_gripper_clt_;

  grasp_learning::SetRBFN set_RBFN_srv_;
  grasp_learning::PolicySearch policy_search_srv_;
  grasp_learning::GetNetworkWeights get_network_weights_srv_;

  std_srvs::Empty empty_srv_;

  /// Servers
  ros::ServiceServer start_demo_srv_;
  ros::ServiceServer run_demo_srv_;
  ros::ServiceServer pause_demo_srv_;
  ros::ServiceServer picture_mode_srv_;
  ros::ServiceServer set_policy_converged_srv_;

  //Publisher oublishing an empty message when the demo grasp is over
  ros::Publisher start_recording_;
  ros::Publisher finish_recording_;
  ros::Publisher run_new_episode_;


  yumi_hw::YumiGrasp grasp_msg;

  grasp_learning::StartRecording start_msg_;
  grasp_learning::FinishRecording finish_msg_;
  //Empty message published by finished_grasping_
  std_msgs::Empty empty_msg_;
  //** Manipulator joint configuration prior to reach-to-grasp */
  std::vector<double> sensing_config_;

  std::vector<std::vector<double>> gripperPos;
  std::vector<std::vector<double>> jointVel;
  std::vector<double> samplingTime;

  std::vector<double> finalPos;


};

}  // end namespace hqp controllers

