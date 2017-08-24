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
    nh_.param<double>("decay_rate", decay_rate_, 1);
    nh_.param<double>("exec_time", exec_time_, 10);
    nh_.param<double>("manifold_height", manifold_height_, 5);
    nh_.param<double>("manifold_radius", manifold_radius_, 5);
    nh_.param<double>("object_radius", object_radius_, 2.5);
    nh_.getParam("manifold_pos", manifoldPos);
    nh_.getParam("final_pos", finalPos);
    nh_.getParam("PC_of_object", PCofObject);
    nh_.param<bool>("nullspace", nullspace_, true);
    nh_.param<int>("max_num_trials", maxNumTrials_, 10);
    nh_.param<int>("max_rollouts_per_trial", maxRolloutsPerTrial_, 500);


    nh_.param<std::string>("relative_path", relativePath, "asd");


    if (with_gazebo_) ROS_INFO("Grasping experiments running in Gazebo.");

  // register general callbacks
    start_demo_srv_ = nh_.advertiseService("start_demo", &DemoLearnManifold::startDemo, this);

    run_demo_srv_ = nh_.advertiseService("run_demo", &DemoLearnManifold::runDemo, this);

    pause_demo_srv_ = nh_.advertiseService("pause_demo", &DemoLearnManifold::pauseDemo, this);

    gripper_pos = nh_.subscribe("robot_state", 2000, &DemoLearnManifold::robotStateCallback, this);
    
    robot_collision = n_.subscribe("/yumi/collision", 10, &DemoLearnManifold::robotCollisionCallback, this);


    set_gazebo_physics_clt_ = n_.serviceClient<gazebo_msgs::SetPhysicsProperties>(
      "/gazebo/set_physics_properties");

    policy_search_clt_ = n_.serviceClient<grasp_learning::PolicySearch>("/RBFNetwork/policy_search");
    add_noise_clt_ = n_.serviceClient<std_srvs::Empty>("/RBFNetwork/add_weight_noise");
    set_RBFN_clt_ = n_.serviceClient<grasp_learning::SetRBFN>("/RBFNetwork/build_RBFNetwork");
    get_network_weights_clt_ = n_.serviceClient<grasp_learning::GetNetworkWeights>("/RBFNetwork/get_running_weights");
    vis_kernel_mean_clt_ = n_.serviceClient<std_srvs::Empty>("/RBFNetwork/visualize_kernel_means");
    reset_RBFN_clt_ = n_.serviceClient<std_srvs::Empty>("/RBFNetwork/reset_RBFN");
    start_demo_clt_ = n_.serviceClient<std_srvs::Empty>("demo_learn_manifold/start_demo");

    start_recording_ = n_.advertise<grasp_learning::StartRecording>("/start_recording",1);
    finish_recording_ = n_.advertise<grasp_learning::FinishRecording>("/finish_recording",1);
    run_new_episode_ = n_.advertise<std_msgs::Empty>("/run_new_episode",1);

    start_msg_.str = ' ';
    finish_msg_.str = ' ';

    if (!with_gazebo_) {
      close_gripper_clt_ = n_.serviceClient<yumi_hw::YumiGrasp>("close_gripper");
      open_gripper_clt_ = n_.serviceClient<yumi_hw::YumiGrasp>("open_gripper");

      close_gripper_clt_.waitForExistence();
      open_gripper_clt_.waitForExistence();
    }
    else{
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
    }
  // PRE-DEFINED JOINT CONFIGURATIONS
  // configs have to be within the safety margins of the joint limits

    sensing_config_ = {-0.42, -1.48, 1.21,  0.75, -0.80, 0.45, 1.21,
     0.42,  -1.48, -1.21, 0.75, 0.80,  0.45, 1.21};

  // DEFAULT GRASP
  grasp_.obj_frame_ = "world";         // object frame
  grasp_.e_frame_ = "gripper_r_base";  // sizeeffector frame
  grasp_.e_.setZero();  // sizeeffector point expressed in the sizeeffector frame
  grasp_.isSphereGrasp = false;
  grasp_.isDefaultGrasp = false;

  createFiles(relativePath);


  setRBFNetwork();
  ROS_INFO("DEMO LEARNING READY.");

}

void DemoLearnManifold::safeShutdown() {
  hiqp_client_.resetHiQPController();
  ROS_BREAK();  // I must break you ... ros::shutdown() doesn't seem to do the
                // job
}

bool DemoLearnManifold::createFiles(std::string relPath){


  std::string trial_folder = relPath + "trial_" + std::to_string(numTrial_)+ "/";
  std::string rew_folder = relPath + "trial_" + std::to_string(numTrial_)+ "/reward/";
  std::string robot_folder = relPath + "trial_" + std::to_string(numTrial_)+ "/robot/";


  std::vector<std::string> folders {trial_folder, rew_folder, robot_folder};

  if(!fileHandler_.createFolders(folders)){
    ROS_INFO("Folders not created");
    return false;
  }

  rewardFile = rew_folder+"rewards.txt";
  
  finalRewardFile = rew_folder+"final_rewards.txt";
  
  normalizedRewardFile = rew_folder + "normalized_rewards.txt";

  pointToLineDistFile = robot_folder+"point_to_line_dist.txt";
  pointToPointDistFile = robot_folder+"point_to_point_dist.txt";


  jointVelFile = robot_folder+"joint_velocity_sum.txt";
  jointVelMessageFile = robot_folder+"joint_vel.txt";
  jointVelAccFile = robot_folder+"joint_vel_acc.txt";

  jointTrajLengthFile = robot_folder+"joint_trajectory_sum.txt";
  jointTrajAccFile = robot_folder+"joint_trajectory_acc.txt";

  gripperPosFile = robot_folder+"gripper_position.txt";
  
  trialDataFile = relPath + "trial_" + std::to_string(numTrial_)+"/trial_data.txt";
  // samplingTimeFile = robot_folder+"sampling_time.txt";

  fileHandler_.createFile(rewardFile);
  fileHandler_.createFile(finalRewardFile);
  fileHandler_.createFile(normalizedRewardFile);

  fileHandler_.createFile(pointToLineDistFile);
  fileHandler_.createFile(pointToPointDistFile);

  fileHandler_.createFile(jointVelFile);
  fileHandler_.createFile(jointVelMessageFile);
  fileHandler_.createFile(jointVelAccFile);

  fileHandler_.createFile(jointTrajLengthFile);
  fileHandler_.createFile(jointTrajAccFile);
  
  fileHandler_.createFile(gripperPosFile);
  
  fileHandler_.createFile(trialDataFile);

  // fileHandler_.createFile(samplingTimeFile);

  return true;
}


    template<typename T>
void DemoLearnManifold::saveDataToFile(std::string file, T data, bool append){
  bool success;
  if (append){
    success = fileHandler_.appendToFile(file, data);
  }
  else{
    success = fileHandler_.writeToFile(file, data);
  }
  if (!success){
    ROS_INFO("Could not store data in file %s",file.c_str());
  }
}

void DemoLearnManifold::safeReset() { hiqp_client_.resetHiQPController(); }

void DemoLearnManifold::robotStateCallback(const grasp_learning::RobotState::ConstPtr& msg){
  gripperPos.push_back(msg->gripperPos);
  jointVel.push_back(msg->jointVel);
  samplingTime.push_back(msg->samplingTime);
}

void DemoLearnManifold::robotCollisionCallback(const std_msgs::Empty::ConstPtr& msg){
  collision_ = true;
}

void DemoLearnManifold::resetTrial(){
  if (reset_RBFN_clt_.call(empty_srv_)){
    ROS_INFO("RBFN successfully resetted");
  }
  else{
    ROS_INFO("Failed to reset RBFN");
  }
  saveTrialDataToFile();
  numTrial_++;
  numRollouts_ = 0;
  collision_ = false;
  policyConverged_ = false;
  init = true;
  createFiles(relativePath);
}

void DemoLearnManifold::saveTrialDataToFile(){

  std::string trialData = "Number of rollouts: "+std::to_string(numRollouts_)+
  "\nCollision: "+std::to_string(collision_) +
  "\nConverged: "+std::to_string(policyConverged_);

  saveDataToFile(trialDataFile, trialData, false);
}

void DemoLearnManifold::updatePolicy(){
  calculateReachingReward();

  // calculateReward();

  ROS_INFO("Calling policy search service");
  if (policy_search_clt_.call(policy_search_srv_)){
    ROS_INFO("Successfully updated policy");
    policyConverged_ = policy_search_srv_.response.converged;
  }
  else{
    ROS_INFO("Failed to update policy");
  }
}

void DemoLearnManifold::calculateReachingReward(){

  std::vector<double> result;

  double Rpos = 1;
  double res = 0;
  std::vector<double> pointToPoint;

  pointToPoint.push_back(pointToPointDist(gripperPos.back(), manifoldPos));

  res = -Rpos*pointToPoint.back();

  result.push_back(exp(res));


  for(unsigned int i=gripperPos.size()-1;i-->0;){
    pointToPoint.push_back(pointToPointDist(gripperPos[i], manifoldPos));
    res -= Rpos*pointToPoint.back();

    result.push_back(result.back()+exp(res));
  }

  std::reverse(result.begin(),result.end());
  std::reverse(pointToPoint.begin(),pointToPoint.end());

  std::vector<double> normalizedRes = normalizeVector(result);


  saveDataToFile(rewardFile, convertToEigenVector(result).transpose(), true);
  saveDataToFile(finalRewardFile, result[0], true);
  saveDataToFile(normalizedRewardFile, convertToEigenVector(normalizedRes), true);


  saveDataToFile(pointToPointDistFile, convertToEigenVector(pointToPoint).transpose(), true);
  
  saveDataToFile(gripperPosFile, convertToEigenMatrix(gripperPos).transpose(), true);

  saveDataToFile(samplingTimeFile, convertToEigenVector(samplingTime).transpose(), true);

  policy_search_srv_.request.reward = result[0];
  policy_search_srv_.request.rewards = normalizedRes;

}


void DemoLearnManifold::calculateReward(){

  std::vector<double> result;

  double Rtraj = 0.5;
  double Rvel = 0.0005;
  double Rpos = 700;
  double res = 0;
  double Rgrasp = 1;
  double Rcollision = -1;
  std::vector<double> jointVel_ = calcJointVel();
  std::vector<double> jointTraj = calcJointTraj();
  std::vector<double> pointToLine;

  pointToLine.push_back(pointToLineDist(gripperPos.back(), PCofObject));

  res = -Rpos*pointToLine.back();



  // if(isCollision()){
  //   res += Rcollision;
  // }
  // if(successfulGrasp()){
  //   res += Rgrasp;
  // }

  result.push_back(exp(res));

  std::vector<double> cumSumVel = accumulateVector(jointVel_);
  std::vector<double> cumSumTraj = accumulateVector(jointTraj);

  for(unsigned int i=gripperPos.size()-1;i-->0;){


    pointToLine.push_back(pointToLineDist(gripperPos[i], PCofObject));
    res -= Rtraj*jointTraj[i];
    res -= Rvel*jointVel_[i];

    result.push_back(result.back()+exp(res));
  }

  std::reverse(result.begin(),result.end());
  std::reverse(pointToLine.begin(),pointToLine.end());

  saveDataToFile(rewardFile, convertToEigenVector(result).transpose(), true);
  saveDataToFile(finalRewardFile, result[0], true);

  saveDataToFile(pointToLineDistFile, convertToEigenVector(pointToLine).transpose(), true);
  
  saveDataToFile(jointVelFile, convertToEigenVector(jointVel_).transpose(), true);
  saveDataToFile(jointVelMessageFile, convertToEigenMatrix(jointVel).transpose(), true);
  saveDataToFile(jointVelAccFile, convertToEigenVector(cumSumVel).transpose(), true);

  saveDataToFile(jointTrajLengthFile, convertToEigenVector(jointTraj).transpose(), true);
  saveDataToFile(jointTrajAccFile, convertToEigenVector(cumSumTraj).transpose(), true);

  saveDataToFile(gripperPosFile, convertToEigenMatrix(gripperPos).transpose(), true);

  saveDataToFile(samplingTimeFile, convertToEigenVector(samplingTime).transpose(), true);

  policy_search_srv_.request.rewards = normalizeVector(result);
  policy_search_srv_.request.reward = result[0];
}


bool DemoLearnManifold::successfulGrasp(){
  double success = -1;
  ROS_INFO("Was the grasp successfull (1) or not (0)?");
  std::cin >> success;
  while(success!=0 || success!= 1){
    ROS_INFO("Invalid input");
    ROS_INFO("Enter 1 for successful grap and 0 otherwise");
    std::cin >> success;
  }
  return (success==1 ? true:false);
}

bool DemoLearnManifold::isCollision(){
  double collision = -1;
  ROS_INFO("Did the manipulator collide with the object (1) or not (0)?");
  std::cin >> collision;
  while(collision!=0 || collision!= 1){
    ROS_INFO("Invalid input");
    ROS_INFO("Enter 1 if collision and 0 otherwise");
    std::cin >> collision;
  }
  return (collision==1 ? true:false);

}

double DemoLearnManifold::pointToLineDist(std::vector<double> point, std::vector<double> line){
  Eigen::Vector3d v_hat;
  v_hat<<line[0], line[1], line[2];
  Eigen::Vector3d d;
  d<<line[3],line[4],line[5];
  Eigen::Vector3d p;
  p<<point[0],point[1],point[2];

  Eigen::Vector3d x = p - d;
  double s = x.dot(v_hat);

  Eigen::Vector3d proj;
  proj = -x+s*v_hat;
  std::vector<double> p2(proj.data(), proj.data() + proj.rows() * proj.cols());
  return vectorLength(p2);
}

double DemoLearnManifold::vectorLength(const std::vector<double>& vec){
  double diff_square = 0;
  double diff = 0;

  for (int i =0;i<vec.size();i++){
    diff = vec[i];
    diff_square += diff*diff;
  }
  return diff_square;
}


std::vector<double> DemoLearnManifold::normalizeVector(const std::vector<double>& v){

  // double norm = sumVector(v);
  double length = v.size();
  std::vector<double> result;
  for(int i = 0;i<v.size();i++){
    // result.push_back(v[i]/norm);
    result.push_back(v[i]/length);
  }
  return result;
}

std::vector<double> DemoLearnManifold::calcJointVel(){
  std::vector<double> vel;
  double temp = 0.0;
  for(int i =0;i<samplingTime.size();i++){
    for(int j = 0;j<jointVel[0].size();j++){
      temp += fabs(jointVel[i][j]);
    }
    vel.push_back(temp);
    temp = 0.0;
  }
  return vel;
}

std::vector<double> DemoLearnManifold::calcJointTraj(){
  std::vector<double> trajectory;
  double temp = 0;
  for(int i =0;i<samplingTime.size();i++){
    for(int j = 0;j<jointVel[0].size();j++){
      temp += samplingTime[i]*fabs(jointVel[i][j]);
    }
    trajectory.push_back(temp);
    temp = 0;
  }
  return trajectory;
}


double DemoLearnManifold::calcJointVelocityOneTimeStep(unsigned int i){
  double vel = 0;
  for(int j = 0;j<jointVel[0].size();j++){
    vel += abs(jointVel[i][j]);
  }
  return vel;

}


double DemoLearnManifold::calcJointMovementOneTimeStep(unsigned int i){
  double trajectory = 0;
  for(int j = 0;j<jointVel[0].size();j++){
    trajectory += samplingTime[i]*abs(jointVel[i][j]);
  }
  return trajectory;
}

double DemoLearnManifold::pointToPointDist(std::vector<double> point1, std::vector<double> point2){
  double diff_square = 0;
  double diff = 0;
  for (int i =0;i<point1.size();i++){
    diff = point1[i]-point2[i];
    diff_square += diff*diff;
  }
  return diff_square;
}

  template<typename T>
Eigen::Map<Eigen::VectorXd> DemoLearnManifold::convertToEigenVector(std::vector<T> vec){
  T* ptr = &vec[0];
  Eigen::Map<Eigen::VectorXd> eigenVec(ptr, vec.size());
  return eigenVec;
}

  template<typename T>
Eigen::MatrixXd DemoLearnManifold::convertToEigenMatrix(std::vector<std::vector<T>> data)
{
  Eigen::MatrixXd eMatrix(data.size(), data[0].size());
  for (int i = 0; i < data.size(); ++i)
    eMatrix.row(i) = Eigen::VectorXd::Map(&data[i][0], data[0].size());
  return eMatrix;
}


template<typename T>
std::vector<T> DemoLearnManifold::accumulateVector(std::vector<T> vec){
  std::vector<T> result(vec.size());
  std::reverse(vec.begin(),vec.end());
  std::partial_sum (vec.begin(), vec.end(), result.begin());
  std::reverse(result.begin(),result.end());
  return  result;
}

double DemoLearnManifold::sumVector(const std::vector<double>& vec){
  return std::accumulate(vec.begin(), vec.end(), 0.0);
}


void DemoLearnManifold::printVector(const std::vector<double>& vec){
  for (auto& a: vec){
    std::cout<<a<<" ";
  }
  std::cout<<std::endl;
} 


void DemoLearnManifold::resetMatrix(std::vector<std::vector<double>>& matrix){
  std::vector<std::vector<double>> newMatrix;
  matrix = newMatrix;
}

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
  set_RBFN_srv_.request.radius = manifold_radius_;
  set_RBFN_srv_.request.height = manifold_height_;
  std::vector<double> vec {manifoldPos[0],manifoldPos[1],manifoldPos[2]};
  set_RBFN_srv_.request.globalPos = vec;

  ROS_INFO("Calling buld RBFN service");

  if (set_RBFN_clt_.call(set_RBFN_srv_)){
    ROS_INFO("RBFN successfully build");
  }
  else{
    ROS_INFO("Failed to build RBFN");
  }

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

void DemoLearnManifold::visualizeKernels(){
  if (vis_kernel_mean_clt_.call(empty_srv_)){
    ROS_INFO("Kernels successfully visualized");
  }
  else{
    ROS_INFO("Failed to visualize kernels");
  }

}


bool DemoLearnManifold::doGraspAndLiftNullspace() {

  if (!with_gazebo_) {
    if (grasp_.isDefaultGrasp) {
      ROS_WARN("Grasp is default grasp!");
      return false;
    }
  }


  hiqp_msgs::Task gripperToManifold;
  hiqp_msgs::Task gripperToGraspPlane;
  hiqp_msgs::Task gripperAxisToTargetAxis;
  hiqp_msgs::Task gripperAxisAlignedToTargetAxis;
  hiqp_msgs::Task gripperToObject;

  hiqp_msgs::Primitive eef_point;
  // hiqp_msgs::Primitive final_point;
  hiqp_msgs::Primitive object;
  hiqp_msgs::Primitive manifold;
  hiqp_msgs::Primitive gripper_approach_axis;
  hiqp_msgs::Primitive gripper_vertical_axis;
  hiqp_msgs::Primitive grasp_target_axis;
  hiqp_msgs::Primitive grasp_plane;
  hiqp_msgs::Primitive PC_of_object;

  // Define the primitives
  eef_point = hiqp_ros::createPrimitiveMsg(
    "point_eef", "point", grasp_.e_frame_, true, {1, 0, 0, 1},
    {grasp_.e_(0), grasp_.e_(1), grasp_.e_(2)+0.1});

  manifold = hiqp_ros::createPrimitiveMsg(
    "grasp_manifold", "cylinder", "world", true, {1.0, 0.0, 0.0, 0.3},
    {0, 0, 1,manifoldPos[0], manifoldPos[1], manifoldPos[2],
      manifold_radius_,manifold_height_});

  // object = hiqp_ros::createPrimitiveMsg(
  //   "object_manifold", "cylinder", "world", true, {1.0, 0.0, 0.0, 0.3},
  //   {0, 0, 1,manifoldPos[0], manifoldPos[1], manifoldPos[2],
  //     object_radius_,manifold_height_});

  gripper_approach_axis = hiqp_ros::createPrimitiveMsg(
    "gripper_approach_axis", "line", grasp_.e_frame_, true, {0, 0, 1, 1},
    {0, 0, 1, 0, 0, 0.1});

  gripper_vertical_axis = hiqp_ros::createPrimitiveMsg(
    "gripper_vertical_axis", "line", grasp_.e_frame_, true, {0, 0, 1, 1},
    {0, -1, 0, 0, 0, 0.1});

  grasp_target_axis = hiqp_ros::createPrimitiveMsg(
    "grasp_target_axis", "line", "world", true, {0, 1, 0, 1},
    {0, 0, 1, manifoldPos[0], manifoldPos[1], manifoldPos[2]});

  grasp_plane = hiqp_ros::createPrimitiveMsg(
    "grasp_plane", "plane", "world", true, {0, 1.0, 0, 0.4},
    {0, 0, 1,manifoldPos[2]+manifold_height_/2});//0.89

  PC_of_object = hiqp_ros::createPrimitiveMsg(
    "PC_of_object", "line", "world", true, {1, 1, 1, 1},
    PCofObject);
  // Define tasks

  gripperToGraspPlane = hiqp_ros::createTaskMsg(
    "gripper_ee_point_on_grasp_plane", 2, true, false, true,
    {"TDefGeomProj", "point", "plane",
    eef_point.name + " = " + grasp_plane.name},
    {"TDynLinear", std::to_string(decay_rate_ * DYNAMICS_GAIN)});

  gripperAxisToTargetAxis = hiqp_ros::createTaskMsg(
    "gripper_approach_axis_coplanar_grasp_target_axis", 2, true, false, true,
    {"TDefGeomProj", "line", "line",
    gripper_approach_axis.name + " = " + grasp_target_axis.name},
    {"TDynLinear", std::to_string(decay_rate_ * DYNAMICS_GAIN)});

  gripperAxisAlignedToTargetAxis = hiqp_ros::createTaskMsg(
    "gripper_vertical_axis_parallel_grasp_target_axis", 2, true, false, true,
    {"TDefGeomAlign", "line", "line",
    gripper_vertical_axis.name + " = " + grasp_target_axis.name,  "0"},
    {"TDynLinear", std::to_string(decay_rate_ * DYNAMICS_GAIN)});

  gripperToManifold = hiqp_ros::createTaskMsg(
    "point_to_manifold", 2, true, false, true,
    {"TDefGeomProjWithNullspace", "point", "cylinder",
    eef_point.name + " = " + manifold.name},
    {"TDynRBFN", std::to_string(decay_rate_ * DYNAMICS_GAIN), std::to_string(false)}); 

  // gripperToObject = hiqp_ros::createTaskMsg(
  //   "point_to_object", 2, true, false, true,
  //   {"TDefGeomProj", "point", "cylinder",
  //   eef_point.name + " = " + object.name},
  //   {"TDynLinear", std::to_string(decay_rate_ * DYNAMICS_GAIN)}); 

  start_recording_.publish(start_msg_);

  using hiqp_ros::TaskDoneReaction;

  hiqp_client_.setPrimitives({eef_point, gripper_approach_axis, gripper_vertical_axis, grasp_target_axis, grasp_plane, manifold, PC_of_object});

  hiqp_client_.setTasks({gripperToGraspPlane, gripperAxisToTargetAxis, gripperToManifold, gripperAxisAlignedToTargetAxis});

  // Activate all tasks for bringing the gripper to the manifold
  hiqp_client_.activateTasks({gripperToGraspPlane.name, gripperAxisToTargetAxis.name,
    gripperToManifold.name, gripperAxisAlignedToTargetAxis.name});

  // Set the gripper approach pose
  hiqp_client_.waitForCompletion(
    {gripperToGraspPlane.name, gripperAxisToTargetAxis.name,
      gripperToManifold.name, gripperAxisAlignedToTargetAxis.name},
      {TaskDoneReaction::REMOVE, TaskDoneReaction::REMOVE, TaskDoneReaction::REMOVE, TaskDoneReaction::REMOVE},
      {1e-10,1e-10,1e-10,1e-10}, exec_time_);


  // hiqp_client_.setTasks({gripperToGraspPlane, gripperAxisToTargetAxis, gripperToObject, gripperAxisAlignedToTargetAxis});

  // Activate all tasks for bringing the gripper to the object manifold
  // hiqp_client_.activateTasks({gripperToGraspPlane.name, gripperAxisToTargetAxis.name,
  //   gripperToObject.name, gripperAxisAlignedToTargetAxis.name});


  // hiqp_client_.waitForCompletion(
  //   {gripperToGraspPlane.name, gripperAxisToTargetAxis.name,
  //     gripperToObject.name, gripperAxisAlignedToTargetAxis.name},
  //     {TaskDoneReaction::REMOVE, TaskDoneReaction::REMOVE, TaskDoneReaction::REMOVE, TaskDoneReaction::REMOVE},
  //     {1e-5,1e-5,1e-10,1e-5}, exec_time_);

  hiqp_client_.removePrimitives({eef_point.name, gripper_approach_axis.name, gripper_vertical_axis.name,
    grasp_target_axis.name, grasp_plane.name, manifold.name, PC_of_object.name});

  finish_recording_.publish(finish_msg_);

  return true;
}

bool DemoLearnManifold::doGraspAndLiftTaskspace() {


  hiqp_msgs::Task frameToFrame;
  hiqp_msgs::Task gripperAxisAlignedToTargetAxis;

  hiqp_msgs::Primitive eef_point;
  hiqp_msgs::Primitive final_point;

  hiqp_msgs::Primitive gripperFrame;
  hiqp_msgs::Primitive pointFrame;
  hiqp_msgs::Primitive PC_of_object;

  eef_point = hiqp_ros::createPrimitiveMsg(
    "point_eef", "point", grasp_.e_frame_, true, {1, 0, 0, 1},
    {grasp_.e_(0), grasp_.e_(1), grasp_.e_(2)+0.1});

  // Define the primitives
  gripperFrame = hiqp_ros::createPrimitiveMsg(
    "gripper_frame", "frame", grasp_.e_frame_, true, {1, 0, 0, 1},
    {grasp_.e_(0), grasp_.e_(1), grasp_.e_(2)+0.1});

  pointFrame = hiqp_ros::createPrimitiveMsg(
    "point_frame", "frame", "world", true, {0.0, 0.0, 1.0, 1},
    {-0.1,0,0});

  final_point = hiqp_ros::createPrimitiveMsg(
    "final_point", "point", "world", true, {1, 1, 0, 1},
    {manifoldPos[0], manifoldPos[1], manifoldPos[2]});

  // Define tasks

  frameToFrame = hiqp_ros::createTaskMsg(
    "frame_to_frame", 2, true, true, true,
    {"TDefGeomProj", "frame", "frame",
    gripperFrame.name + " = " + pointFrame.name},
    {"TDynRBFN", std::to_string(10*decay_rate_ * DYNAMICS_GAIN), std::to_string(true)});

  // frameToFrame = hiqp_ros::createTaskMsg(
  //   "frame_to_frame", 2, true, true, true,
  //   {"TDefGeomProj", "frame", "frame",
  //   gripperFrame.name + " = " + pointFrame.name},
  //   {"TDynLinear", std::to_string(10*decay_rate_ * DYNAMICS_GAIN)});

  start_recording_.publish(start_msg_);

  using hiqp_ros::TaskDoneReaction;

  hiqp_client_.setPrimitives({eef_point, gripperFrame, pointFrame, final_point});

  hiqp_client_.setTasks({frameToFrame});

  // Activate all tasks for bringing the gripper to the manifold
  // hiqp_client_.activateTasks({frameToFrame.name});

  // Set the gripper approach pose
  hiqp_client_.waitForCompletion(
    {frameToFrame.name},
    {TaskDoneReaction::REMOVE},
    {0}, exec_time_);


  hiqp_client_.removePrimitives({eef_point.name, gripperFrame.name, pointFrame.name, final_point.name});

  finish_recording_.publish(finish_msg_);

  return true;
}

bool DemoLearnManifold::pauseDemo(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res){
  run_demo_=false;
}

bool DemoLearnManifold::runDemo(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res){

  run_demo_=true;
  while(run_demo_){
    if(numTrial_<=maxNumTrials_){
      ROS_INFO("Trial number %d.", numTrial_);
      while(run_demo_){
        if(numRollouts_<=maxRolloutsPerTrial_){
          ROS_INFO("Starting rollout number %d.", numRollouts_);
          if(!start_demo_clt_.call(empty_srv_)){
            ROS_ERROR("Failed to run demo.");
            return false;
          }
          if(collision_ && !init){
            ROS_INFO("Trial failed due to collision with the environment after %d rollouts.\nResetting everything and starting a new trial.", numRollouts_);
            resetTrial();
            break;
          }
          else if(policyConverged_ && !init){
            ROS_INFO("Trial suceeded as the policy converged after %d rollouts.\nResetting everything and starting a new trial.", numRollouts_);
            resetTrial();
            break;
          }
        }
        else{
          ROS_INFO("The maximum number of %d rollouts has been reached.\nResetting everything and starting a new trial.", maxRolloutsPerTrial_);
          resetTrial();
          break;
        }
      }
    }
    else{
      ROS_INFO("The maximum number of %d trials has been reached.\nNot performing any more trials.", maxNumTrials_);
      break;
    }
  }
  return true;
}



bool DemoLearnManifold::startDemo(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res) {

  ROS_INFO("Starting demo");
  hiqp_client_.resetHiQPController();

  // MANIPULATOR SENSING CONFIGURATION
  visualizeKernels();
  hiqp_client_.setJointAngles(sensing_config_);

  // Empty vector that contains gripper position
  resetMatrix(gripperPos);
  resetMatrix(jointVel);
  samplingTime.clear();
  // GRASP APPROACH


  ROS_INFO("Trying grasp approach.");

  addNoise();

  if(nullspace_){
    if (!doGraspAndLiftNullspace()) {
      ROS_ERROR("Could not set the grasp pose!");
      safeShutdown();
      return false;
    }
  }
  else{
    if (!doGraspAndLiftTaskspace()) {
      ROS_ERROR("Could not set the grasp pose!");
      safeShutdown();
      return false;
    }
  }

  if (!init){
    if(!collision_){
      ROS_INFO("Updating policy");
      updatePolicy();
      numRollouts_++;
    }
  }
  else{
    init = false;
  }

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