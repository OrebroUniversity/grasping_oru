#include <grasp_learning/demo_learn_manifold.h>


namespace demo_learning {
  using namespace boost::assign;
//-----------------------------------------------------------------
  DemoLearnManifold::DemoLearnManifold()
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
    nh_.param<int>("num_kernels", num_kernels_, 50);
    nh_.param<int>("num_kernel_rows", num_kernel_rows_, 1);
    nh_.param<double>("manifold_height", manifold_height_, 5);
    nh_.param<double>("manifold_radius", manifold_radius_, 5);
    nh_.getParam("manifold_pos", manifoldPos);
    nh_.getParam("final_pos", finalPos);

    nh_.param<int>("burn_in_trials",  burn_in_trials_, 8);
    nh_.param<int>("max_num_samples", max_num_samples_, 8);

    nh_.param<std::string>("task", task_name_, "gripperToHorizontalPlane");

    if (with_gazebo_) ROS_INFO("Grasping experiments running in Gazebo.");

  // register general callbacks
    start_demo_srv_ =
    nh_.advertiseService("start_demo", &DemoLearnManifold::startDemo, this);

    gripper_pos = nh_.subscribe("gripper_pos", 1000, &DemoLearnManifold::gripperPosCallback, this);
    
    set_gazebo_physics_clt_ = n_.serviceClient<gazebo_msgs::SetPhysicsProperties>(
      "/gazebo/set_physics_properties");

    policy_search_clt_ = n_.serviceClient<grasp_learning::PolicySearch>("/policy_search");
    add_noise_clt_ = n_.serviceClient<std_srvs::Empty>("/add_weight_noise");
    set_RBFN_clt_ = n_.serviceClient<grasp_learning::SetRBFN>("/build_RBFNetwork");
    get_network_weights_clt_ = n_.serviceClient<grasp_learning::GetNetworkWeights>("/get_running_weights");

    start_msg_.str = ' ';
    finish_msg_.str = ' ';
    start_recording_ = n_.advertise<grasp_learning::StartRecording>("/start_recording",1);
    finish_recording_ = n_.advertise<grasp_learning::FinishRecording>("/finish_recording",1);
    run_new_episode_ = n_.advertise<std_msgs::Empty>("/run_new_episode",1);


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
    } else{
      ROS_INFO("Disabled gravity in Gazebo.");
    }
  // PRE-DEFINED JOINT CONFIGURATIONS
  // configs have to be within the safety margins of the joint limits

    sensing_config_ = {-0.42, -1.48, 1.21,  0.75, -0.80, 0.45, 1.21,
     0.42,  -1.48, -1.21, 0.75, 0.80,  0.45, 1.21};

  // DEFAULT GRASP

  grasp_upper_.obj_frame_ = "world";         // object frame
  grasp_upper_.e_frame_ = "gripper_r_base";  // sizeeffector frame
  grasp_upper_.e_.setZero();  // sizeeffector point expressed in the sizeeffector frame
  grasp_upper_.isSphereGrasp = false;
  grasp_upper_.isDefaultGrasp = true;

  grasp_lower_.obj_frame_ = "world";         // object frame
  grasp_lower_.e_frame_ = "gripper_r_base";  // sizeeffector frame
  grasp_lower_.e_.setZero();  // sizeeffector point expressed in the sizeeffector frame
  grasp_lower_.isSphereGrasp = false;
  grasp_lower_.isDefaultGrasp = true;

  setRBFNetwork();
  ROS_INFO("DEMO LEARNING READY.");

}

void DemoLearnManifold::safeShutdown() {
  hiqp_client_.resetHiQPController();
  ROS_BREAK();  // I must break you ... ros::shutdown() doesn't seem to do the
                // job
}

void DemoLearnManifold::safeReset() { hiqp_client_.resetHiQPController(); }

void DemoLearnManifold::gripperPosCallback(const std_msgs::Float64MultiArray::ConstPtr& msg){
  gripperPos.push_back(msg->data);
}

void DemoLearnManifold::updatePolicy(){
  policy_search_srv_.request.reward = calculateReward();
  ROS_INFO("Calling policy search service");
  if (policy_search_clt_.call(policy_search_srv_)){
    ROS_INFO("Successfully updated policy");
  }
  else{
    ROS_INFO("Failed to update policy");
  }
}

double DemoLearnManifold::calculateReward(){
  double result = 0;
  get_network_weights_srv_.request.str = ' ';
  get_network_weights_clt_.call(get_network_weights_srv_);
  std::vector<double> weights {get_network_weights_srv_.response.weights};
  result = dotProduct(weights);
  result += 10*pointToPointDist(gripperPos.back(), finalPos);
  ROS_INFO("Reward is %lf", exp(-result));
  return exp(-result);
}

double DemoLearnManifold::dotProduct(std::vector<double> vec){
  double res = 0;
  for(auto& iter: vec){
    res += iter*iter;
  }
  return res;
}

double DemoLearnManifold::pointToPointDist(std::vector<double> point1, std::vector<double> point2){
  double diff_square = 0;
  double diff = 0;
  for (int i =0;i<point1.size();i++){
    diff = point1[i]-point2[i];
    diff_square += diff*diff;
  }
  return std::sqrt(diff_square);
}

// double DemoLearnManifold::pointToLineDist(std::string point, std::string line){


// }

void DemoLearnManifold::addNoise(){
  ROS_INFO("Calling add noise service");
  if (add_noise_clt_.call(empty_srv_)){
    ROS_INFO("Added noise successfully");
  }
  else{
    ROS_INFO("Failed to add noise");
  }
}

void DemoLearnManifold::setRBFNetwork(){
  set_RBFN_srv_.request.numKernels = num_kernels_;
  set_RBFN_srv_.request.numRows = num_kernel_rows_;
  set_RBFN_srv_.request.radius = manifold_radius_;
  set_RBFN_srv_.request.height = manifold_height_;
  set_RBFN_srv_.request.globalPos = manifoldPos;
  
  ROS_INFO("Calling buld RBFN service");

  if (set_RBFN_clt_.call(set_RBFN_srv_)){
    ROS_INFO("RBFN successfully build");
  }
  else{
    ROS_INFO("Failed to build RBFN");
  }

}

void DemoLearnManifold::printVector(const std::vector<double>& vec){
  for (auto& a: vec){
    std::cout<<a<<" ";
  }
  std::cout<<std::endl;
} 

std::vector<double> DemoLearnManifold::generateStartPosition(std::vector<double> curr_sensing_vec){

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


bool DemoLearnManifold::doGraspAndLift() {

  addNoise();

  hiqp_msgs::Task gripperBelowUpperPlane;
  hiqp_msgs::Task gripperAboveLowerPlane;
  hiqp_msgs::Task gripperTomanifold;
  hiqp_msgs::Task gripperToGraspPlane;
  hiqp_msgs::Task gripperAxisToTargetAxis;
  hiqp_msgs::Task gripperAxisAlignedToTargetAxis;

  hiqp_msgs::Primitive eef_point;
  hiqp_msgs::Primitive final_point;
  hiqp_msgs::Primitive manifold;
  hiqp_msgs::Primitive gripper_approach_axis;
  hiqp_msgs::Primitive gripper_vertical_axis;
  hiqp_msgs::Primitive grasp_target_axis;
  hiqp_msgs::Primitive grasp_plane;

  // Define the primitives
  eef_point = hiqp_ros::createPrimitiveMsg(
    "point_eef", "point", grasp_lower_.e_frame_, true, {1, 0, 0, 1},
    {grasp_lower_.e_(0), grasp_lower_.e_(1), grasp_lower_.e_(2)+0.1});

  final_point = hiqp_ros::createPrimitiveMsg(
    "final_point", "point", "world", true, {0, 0, 1, 1},
    {finalPos[0], finalPos[1], finalPos[2]});

  manifold = hiqp_ros::createPrimitiveMsg(
    "grasp_manifold", "cylinder", "world", true, {1.0, 0.0, 0.0, 0.3},
    {0, 0, 1,manifoldPos[0], manifoldPos[1], manifoldPos[2],
      manifold_radius_,manifold_height_});

  gripper_approach_axis = hiqp_ros::createPrimitiveMsg(
    "gripper_approach_axis", "line", grasp_lower_.e_frame_, true, {0, 0, 1, 1},
    {0, 0, 1, 0, 0, 0.1});

  gripper_vertical_axis = hiqp_ros::createPrimitiveMsg(
    "gripper_vertical_axis", "line", grasp_lower_.e_frame_, true, {0, 0, 1, 1},
    {0, -1, 0, 0, 0, 0.1});

  grasp_target_axis = hiqp_ros::createPrimitiveMsg(
    "grasp_target_axis", "line", "world", true, {0, 1, 0, 1},
    {0, 0, 1, manifoldPos[0], manifoldPos[1], manifoldPos[2]});

  grasp_plane = hiqp_ros::createPrimitiveMsg(
    "grasp_plane", "plane", "world", true, {0, 1.0, 0, 0.4},
    {0, 0, 1,1.05});//0.89

  // Define tasks

  gripperToGraspPlane = hiqp_ros::createTaskMsg(
    "gripper_ee_point_on_grasp_plane", 1, true, false, true,
    {"TDefGeomProj", "point", "plane",
    eef_point.name + " = " + grasp_plane.name},
    {"TDynLinear", std::to_string(decay_rate_ * DYNAMICS_GAIN)});

  gripperAxisToTargetAxis = hiqp_ros::createTaskMsg(
    "gripper_approach_axis_coplanar_grasp_target_axis", 1, true, false, true,
    {"TDefGeomProj", "line", "line",
    gripper_approach_axis.name + " = " + grasp_target_axis.name},
    {"TDynLinear", std::to_string(decay_rate_ * DYNAMICS_GAIN)});

  // gripperTomanifold = hiqp_ros::createTaskMsg(
  //   "gripper_ee_point_on_grasp_cylinder", 2, true, false, true,
  //   {"TDefGeomProj", "point", "cylinder",
  //   eef_point.name + " = " + manifold.name},
  //   {"TDynLinear", std::to_string(decay_rate_ * DYNAMICS_GAIN)});

  gripperAxisAlignedToTargetAxis = hiqp_ros::createTaskMsg(
    "gripper_vertical_axis_parallel_grasp_target_axis", 1, true, false, true,
    {"TDefGeomAlign", "line", "line",
    gripper_vertical_axis.name + " = " + grasp_target_axis.name, "0"},
    {"TDynLinear", std::to_string(decay_rate_ * DYNAMICS_GAIN)});

  // gripperTomanifold = hiqp_ros::createTaskMsg(
  //   "point_to_manifold", 1, true, false, true,
  //   {"TDefGeomProj", "point", "cylinder",
  //   eef_point.name + " = " + manifold.name},
  //   {"TDynRandom", "0", "1"}); 

  gripperTomanifold = hiqp_ros::createTaskMsg(
    "point_to_manifold", 2, true, false, true,
    {"TDefGeomProjWithNullspace", "point", "cylinder",
    eef_point.name + " = " + manifold.name},
    {"TDynRBFN", std::to_string(decay_rate_ * DYNAMICS_GAIN)}); 

  start_recording_.publish(start_msg_);

  using hiqp_ros::TaskDoneReaction;

  // if(initialize){

    // hiqp_client_.setPrimitives({eef_point, gripper_approach_axis, gripper_vertical_axis, grasp_target_axis, grasp_plane, final_point});
    // hiqp_client_.setTasks({gripperToGraspPlane, gripperAxisToTargetAxis, gripperAxisAlignedToTargetAxis});

    hiqp_client_.setPrimitives({eef_point, gripper_approach_axis, gripper_vertical_axis, grasp_target_axis, grasp_plane, manifold, final_point});
    hiqp_client_.setTasks({gripperToGraspPlane, gripperAxisToTargetAxis, gripperTomanifold, gripperAxisAlignedToTargetAxis});


  //   initialize = false;
  // }
  // else{

    // hiqp_client_.setPrimitives({eef_point, gripper_approach_axis, gripper_vertical_axis, grasp_target_axis, grasp_plane, final_point});
    // hiqp_client_.setTasks({gripperToGraspPlane, gripperAxisToTargetAxis, gripperAxisAlignedToTargetAxis});

    // hiqp_client_.setPrimitives({eef_point, gripper_approach_axis, gripper_vertical_axis, grasp_target_axis, grasp_plane, manifold, final_point});
  //   hiqp_client_.setTasks({gripperToGraspPlane, gripperAxisToTargetAxis, gripperTomanifold, gripperAxisAlignedToTargetAxis});

  // }


  // hiqp_client_.activateTasks({gripperToGraspPlane.name, gripperAxisToTargetAxis.name,
  //    gripperAxisAlignedToTargetAxis.name});
  // hiqp_client_.waitForCompletion(
  //   {gripperToGraspPlane.name, gripperAxisToTargetAxis.name,
  //     gripperAxisAlignedToTargetAxis.name},
  //     {TaskDoneReaction::REMOVE, TaskDoneReaction::REMOVE, TaskDoneReaction::REMOVE},
  //     {1e-10,1e-10,1e-10}, exec_time_);

  hiqp_client_.activateTasks({gripperToGraspPlane.name, gripperAxisToTargetAxis.name,
    gripperTomanifold.name, gripperAxisAlignedToTargetAxis.name});
  hiqp_client_.waitForCompletion(
    {gripperToGraspPlane.name, gripperAxisToTargetAxis.name,
      gripperTomanifold.name, gripperAxisAlignedToTargetAxis.name},
      {TaskDoneReaction::REMOVE, TaskDoneReaction::REMOVE, TaskDoneReaction::REMOVE, TaskDoneReaction::REMOVE},
      {1e-10,1e-10,1e-10,1e-10}, exec_time_);

  // hiqp_client_.removePrimitives({eef_point.name, gripper_approach_axis.name, gripper_vertical_axis.name,
  // grasp_target_axis.name, grasp_plane.name, final_point.name});

  hiqp_client_.removePrimitives({eef_point.name, gripper_approach_axis.name, gripper_vertical_axis.name,
  grasp_target_axis.name, grasp_plane.name, final_point.name, manifold.name});
  hiqp_client_.removeTasks({gripperToGraspPlane.name, gripperAxisToTargetAxis.name,
    gripperTomanifold.name, gripperAxisAlignedToTargetAxis.name});

  finish_recording_.publish(finish_msg_);


  // --------------- //
  // --- Extract --- //
  // --------------- //


  return true;
}

void DemoLearnManifold::resetMatrix(std::vector<std::vector<double>>& matrix){
  std::vector<std::vector<double>> newMatrix;
  matrix = newMatrix;
}

bool DemoLearnManifold::startDemo(std_srvs::Empty::Request& req,
 std_srvs::Empty::Response& res) {
  ROS_INFO("Starting demo");
  hiqp_client_.resetHiQPController();


  // MANIPULATOR SENSING CONFIGURATION
  hiqp_client_.setJointAngles(sensing_config_);

  // Empty vector that contains gripper position
  resetMatrix(gripperPos);

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
  ros::init(argc, argv, "demo_learn_manifold");

  demo_learning::DemoLearnManifold demo_learning;

  ROS_INFO("Demo grasping node ready");
  ros::AsyncSpinner spinner(4);  // Use 4 threads
  spinner.start();
  ros::waitForShutdown();

  return 0;
}
