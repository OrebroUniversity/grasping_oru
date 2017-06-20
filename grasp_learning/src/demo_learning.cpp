#include <grasp_learning/demo_learning.h>

#include <gazebo_msgs/SetPhysicsProperties.h>
#include <math.h>
#include <time.h>
#include <boost/assign/std/vector.hpp>
#include <limits>

#include <yumi_hw/YumiGrasp.h>

namespace demo_learning {
  using namespace boost::assign;
//-----------------------------------------------------------------
  DemoLearning::DemoLearning()
  : n_jnts(14), hiqp_client_("yumi", "hiqp_joint_velocity_controller") {
  // handle to home
    nh_ = ros::NodeHandle("~");
  // global handle
    n_ = ros::NodeHandle();

  // get params
    nh_.param<bool>("with_gazebo", with_gazebo_, false);
    nh_.param<bool>("generalize_policy", generalize_policy_, false);
    nh_.param<double>("decay_rate", decay_rate_, 1);
    nh_.param<std::string>("task_dynamics", task_dynamics_, "TDynPolicy");
    nh_.param<double>("exec_time", exec_time_, 10);


    nh_.param<std::string>("task", task_name_, "gripperToHorizontalPlane");

    if (with_gazebo_) ROS_INFO("Grasping experiments running in Gazebo.");

  // register general callbacks
    start_demo_srv_ =
    nh_.advertiseService("start_demo", &DemoLearning::startDemo, this);

    set_gazebo_physics_clt_ = n_.serviceClient<gazebo_msgs::SetPhysicsProperties>(
      "/gazebo/set_physics_properties");

    client_Policy_Search_ = n_.serviceClient<grasp_learning::PolicySearch>("policy_Search");

    start_msg_.str = ' ';
    finish_msg_.str = ' ';
    start_recording_ = n_.advertise<grasp_learning::StartRecording>("/start_recording",1);
    finish_recording_ = n_.advertise<grasp_learning::FinishRecording>("/finish_recording",1);
    run_new_episode_ = n_.advertise<std_msgs::Empty>("/run_new_episode",1);

    sample_And_Reweight_ = n_.advertise<std_msgs::Empty>("/sample_and_rewight",1);

    // if gazebo is used, set the simulated gravity to zero in order to prevent
    // gazebo's joint drifting glitch
      set_gazebo_physics_clt_.waitForExistence();
      gazebo_msgs::SetPhysicsProperties properties;
      properties.request.time_step = 0.001;
      properties.request.max_update_rate = 1000;
      properties.request.gravity.x = 0.0;
      properties.request.gravity.y = 0.0;
      properties.request.gravity.z = 0.0;
      properties.request.ode_config.auto_disable_bodies = false;
      properties.request.ode_config.sor_pgs_precon_iters = 0;
      properties.request.ode_config.sor_pgs_iters = 50;
      properties.request.ode_config.sor_pgs_w = 1.3;
      properties.request.ode_config.sor_pgs_rms_error_tol = 0.0;
      properties.request.ode_config.contact_surface_layer = 0.001;
      properties.request.ode_config.contact_max_correcting_vel = 100.0;
      properties.request.ode_config.cfm = 0.0;
      properties.request.ode_config.erp = 0.2;
      properties.request.ode_config.max_contacts = 20.0;

      set_gazebo_physics_clt_.call(properties);
      if (!properties.response.success) {
        ROS_ERROR("Couldn't set Gazebo physics properties, status message: %s!",
          properties.response.status_message.c_str());
        ros::shutdown();
      } else
      ROS_INFO("Disabled gravity in Gazebo.");
  
  // PRE-DEFINED JOINT CONFIGURATIONS
  // configs have to be within the safety margins of the joint limits

    sensing_config_ = {-0.42, -1.48, 1.21,  0.75, -0.80, 0.45, 1.21,
     0.42,  -1.48, -1.21, 0.75, 0.80,  0.45, 1.21};

  // DEFAULT GRASP
  grasp_horizontal_.obj_frame_ = "world";         // object frame
  grasp_horizontal_.e_frame_ = "gripper_r_base";  // sizeeffector frame
  grasp_horizontal_.e_.setZero();  // sizeeffector point expressed in the sizeeffector frame
  grasp_horizontal_.isSphereGrasp = false;
  grasp_horizontal_.isDefaultGrasp = true;

  grasp_vertical_.obj_frame_ = "world";         // object frame
  grasp_vertical_.e_frame_ = "gripper_r_base";  // sizeeffector frame
  grasp_vertical_.e_.setZero();  // sizeeffector point expressed in the sizeeffector frame
  grasp_vertical_.isSphereGrasp = false;
  grasp_vertical_.isDefaultGrasp = true;

  std::normal_distribution<double> d2(0,0.1);
  dist.param(d2.param());

      ROS_INFO("DEMO LEARNING READY.");

}

void DemoLearning::safeShutdown() {
  hiqp_client_.resetHiQPController();
  ROS_BREAK();  // I must break you ... ros::shutdown() doesn't seem to do the
                // job
}

void DemoLearning::safeReset() { hiqp_client_.resetHiQPController(); }


void DemoLearning::updatePolicy(){

  grasp_learning::PolicySearch srv_;

  srv_.request.str = " ";  //yumi_joint_1_r


  ROS_INFO("Calling update node");
  if (client_Policy_Search_.call(srv_)){
    ROS_INFO("Policy successfully updated");
    converged_policy_ = srv_.response.converged;
  }
  else{
   ROS_INFO("Could not update policy"); 
 }
}

void DemoLearning::printVector(const std::vector<double>& vec){
  for (auto& a: vec){
    std::cout<<a<<" ";
  }
  std::cout<<std::endl;
} 

std::vector<double> DemoLearning::generateStartPosition(std::vector<double> curr_sensing_vec){

  std::vector<double> new_sensing_config = curr_sensing_vec;
  for(unsigned int i=0;i<7;i++){
    new_sensing_config[i] += this->dist(this->generator);
  }
  ROS_INFO("Previous start configuration");
  printVector(sensing_config_);
  ROS_INFO("New start configuration");
  printVector(new_sensing_config);

  return new_sensing_config;

}


bool DemoLearning::doGraspAndLift() {

  hiqp_msgs::Task gripperToHorizontalPlane;
  hiqp_msgs::Task gripperToPlane;
  hiqp_msgs::Task gripperToVerticalPlane;
  hiqp_msgs::Task gripperToHorizontalAndVerticalPlane;
  hiqp_msgs::Primitive eef_point;

  // Some assertions to make sure that the grasping constraints are alright.
  // TODO: Write this.

  // Primitive for the size effector point.
  eef_point = hiqp_ros::createPrimitiveMsg(
    "point_eef", "point", grasp_horizontal_.e_frame_, true, {1, 0, 0, 1},
    {grasp_horizontal_.e_(0), grasp_horizontal_.e_(1), grasp_horizontal_.e_(2)+0.1});

  // Primitive for the size effector point.

  ROS_INFO("Has the policy converged %d", converged_policy_);


  if (generalize_policy_ && converged_policy_){

    double delta_z = 1.3+this->dist(this->generator);
    ROS_INFO("Table height is now  %lf", delta_z);

    grasp_horizontal_.plane = hiqp_ros::createPrimitiveMsg(
      "table_plane", "plane", "world", true, {0, 1, 0, 0.2},
    {0, 0, 1, delta_z});//0.89


  }
  else{
   grasp_horizontal_.plane = hiqp_ros::createPrimitiveMsg(
    "table_plane_horizontal", "plane", "world", true, {0, 1, 0, 0.4},
    {0, 0, 1,0.2});//0.89

   grasp_vertical_.plane = hiqp_ros::createPrimitiveMsg(
    "table_plane_vertical", "plane", "world", true, {0, 0, 1, 0.4},
    {0, -1, 0,0});//0.89


 }
  // Define tasks

 gripperToHorizontalAndVerticalPlane = hiqp_ros::createTaskMsg(
  "point_to_2_planes", 1, true, true, true,
  {"TDefMetaTask", "TDefGeomProj", "point", "plane",
  eef_point.name + " = " + grasp_horizontal_.plane.name,
  "TDefGeomProj", "point", "plane",
  eef_point.name + " = " + grasp_vertical_.plane.name},
  {task_dynamics_, std::to_string(decay_rate_ * DYNAMICS_GAIN)});

 gripperToHorizontalPlane = hiqp_ros::createTaskMsg(
  "point_to_horizontal_plane", 1, true, true, true,
  {"TDefMetaTask", "TDefGeomProj", "point", "plane",
  eef_point.name + " = " + grasp_horizontal_.plane.name},
  {task_dynamics_,std::to_string(decay_rate_ * DYNAMICS_GAIN)});

 gripperToVerticalPlane = hiqp_ros::createTaskMsg(
  "point_to_vertical_plane", 1, true, true, true,
  {"TDefGeomProj", "point", "plane",
  eef_point.name + " = " + grasp_vertical_.plane.name},
  {task_dynamics_, std::to_string(decay_rate_ * DYNAMICS_GAIN)});

 start_recording_.publish(start_msg_);

 using hiqp_ros::TaskDoneReaction;


 if(task_name_.compare("gripperToHorizontalPlane")==0){
   hiqp_client_.setPrimitives({eef_point, grasp_horizontal_.plane});
   hiqp_client_.setTasks({gripperToHorizontalPlane});
   hiqp_client_.waitForCompletion(
    {gripperToHorizontalPlane.name},
    {TaskDoneReaction::REMOVE},
    {1e-10}, exec_time_);
   hiqp_client_.removePrimitives({eef_point.name, grasp_horizontal_.plane.name});
 }
 else if(task_name_.compare("gripperToVerticalPlane")==0){
   hiqp_client_.setPrimitives({eef_point, grasp_vertical_.plane});
   hiqp_client_.setTasks({gripperToVerticalPlane});
   hiqp_client_.waitForCompletion(
    {gripperToVerticalPlane.name},
    {TaskDoneReaction::REMOVE},
    {1e-10}, exec_time_);
   hiqp_client_.removePrimitives({eef_point.name, grasp_vertical_.plane.name});
 }
 else if(task_name_.compare("gripperToHorizontalAndVerticalPlane")==0){
   hiqp_client_.setPrimitives({eef_point, grasp_horizontal_.plane, grasp_vertical_.plane});
   hiqp_client_.setTasks({gripperToHorizontalAndVerticalPlane});
   hiqp_client_.waitForCompletion(
    {gripperToHorizontalAndVerticalPlane.name},
    {TaskDoneReaction::REMOVE},
    {1e-10}, exec_time_);
   hiqp_client_.removePrimitives({eef_point.name, grasp_horizontal_.plane.name, grasp_vertical_.plane.name});
 }

 finish_recording_.publish(finish_msg_);


  // --------------- //
  // --- Extract --- //
  // --------------- //


 return true;
}


bool DemoLearning::startDemo(std_srvs::Empty::Request& req,
 std_srvs::Empty::Response& res) {
  ROS_INFO("Starting demo");
  hiqp_client_.resetHiQPController();

  // MANIPULATOR SENSING CONFIGURATION
  hiqp_client_.setJointAngles(sensing_config_);

  // TODO: Detect Stagnation.
  // double accumulated_recent_rewards = prevRewards(reward_vec_,10);

  // if (accumulated_recent_rewards < reward_treshold_){
  sample_And_Reweight_.publish(empty_msg_);
  // }


  // GRASP APPROACH
  ROS_INFO("Trying grasp approach.");

  if (!doGraspAndLift()) {
    ROS_ERROR("Could not set the grasp approach!");
    safeShutdown();
    return false;
  }


  ROS_INFO("Updating policy");
  updatePolicy();


  ROS_INFO("Trying to put the manipulator in transfer configuration.");
  hiqp_client_.setJointAngles(sensing_config_);

  ROS_INFO("DEMO FINISHED.");

  run_new_episode_.publish(empty_msg_);

  return true;
}



}  // size namespace demo_grasping

int main(int argc, char** argv) {
  ros::init(argc, argv, "demo_learning");

  demo_learning::DemoLearning demo_learning;

  ROS_INFO("Demo grasping node ready");
  ros::AsyncSpinner spinner(4);  // Use 4 threads
  spinner.start();
  ros::waitForShutdown();

  return 0;
}
