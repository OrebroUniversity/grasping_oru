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
    nh_.getParam("manifold_pos", manifoldPos);
    nh_.getParam("final_pos", finalPos);
    nh_.getParam("PC_of_object", PCofObject);
    nh_.param<std::string>("relative_path", relativePath, "asd");

    if (with_gazebo_) ROS_INFO("Grasping experiments running in Gazebo.");

  // register general callbacks
    start_demo_srv_ =
    nh_.advertiseService("start_demo", &DemoLearnManifold::startDemo, this);

    gripper_pos = nh_.subscribe("robot_state", 2000, &DemoLearnManifold::robotStateCallback, this);
    
    set_gazebo_physics_clt_ = n_.serviceClient<gazebo_msgs::SetPhysicsProperties>(
      "/gazebo/set_physics_properties");

    policy_search_clt_ = n_.serviceClient<grasp_learning::PolicySearch>("/RBFNetwork/policy_search");
    add_noise_clt_ = n_.serviceClient<std_srvs::Empty>("/RBFNetwork/add_weight_noise");
    set_RBFN_clt_ = n_.serviceClient<grasp_learning::SetRBFN>("/RBFNetwork/build_RBFNetwork");
    get_network_weights_clt_ = n_.serviceClient<grasp_learning::GetNetworkWeights>("/RBFNetwork/get_running_weights");
    vis_kernel_mean_clt_ = n_.serviceClient<std_srvs::Empty>("/RBFNetwork/visualize_kernel_means");

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
  grasp_.obj_frame_ = "world";         // object frame
  grasp_.e_frame_ = "gripper_r_base";  // sizeeffector frame
  grasp_.e_.setZero();  // sizeeffector point expressed in the sizeeffector frame
  grasp_.isSphereGrasp = false;
  grasp_.isDefaultGrasp = true;

  rewardFile = relativePath + "reward/rewards.txt";

  
  finalRewardFile = relativePath + "reward/final_rewards.txt";
  
  pointToLineDistFile = relativePath + "robot/point_to_line_dist.txt";
  
  jointVelFile = relativePath + "robot/joint_velocity_sum.txt";
  jointVelMessageFile = relativePath + "robot/joint_vel.txt";
  jointVelAccFile = relativePath + "robot/joint_vel_acc.txt";

  jointTrajLengthFile = relativePath + "robot/joint_trajectory_sum.txt";
  jointTrajAccFile = relativePath + "robot/joint_trajectory_acc.txt";


  gripperPosFile = relativePath + "robot/gripper_position.txt";
  
  samplingTimeFile = relativePath + "robot/sampling_time.txt";


  fileHandler_.createFile(rewardFile);
  fileHandler_.createFile(finalRewardFile);

  fileHandler_.createFile(pointToLineDistFile);

  fileHandler_.createFile(jointVelFile);
  fileHandler_.createFile(jointVelMessageFile);
  fileHandler_.createFile(jointVelAccFile);


  fileHandler_.createFile(jointTrajLengthFile);
  fileHandler_.createFile(jointTrajAccFile);
  
  fileHandler_.createFile(gripperPosFile);
  
  fileHandler_.createFile(samplingTimeFile);


  setRBFNetwork();
  ROS_INFO("DEMO LEARNING READY.");

}

void DemoLearnManifold::safeShutdown() {
  hiqp_client_.resetHiQPController();
  ROS_BREAK();  // I must break you ... ros::shutdown() doesn't seem to do the
                // job
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

void DemoLearnManifold::updatePolicy(){
  calculateReward();

  ROS_INFO("Calling policy search service");
  if (policy_search_clt_.call(policy_search_srv_)){
    ROS_INFO("Successfully updated policy");
  }
  else{
    ROS_INFO("Failed to update policy");
  }
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
  res = -Rpos*pointToLine.back();

  std::vector<double> cumSumVel = accumulateVector(jointVel_);
  std::vector<double> cumSumTraj = accumulateVector(jointTraj);

  for(unsigned int i=gripperPos.size()-1;i-->0;){
    pointToLine.push_back(pointToLineDist(gripperPos[i], PCofObject));
    res -= Rtraj*jointTraj[i];
    res -= Rvel*jointVel_[i];

    result.push_back(result.back()+exp(res));
  }



  // ROS_INFO("\n\n");
  // ROS_INFO("Trajectory length is %lf", Rtraj*totalJointTraj);
  // ROS_INFO("Sum of all joint velociteis %lf", Rvel*totalJointVel);
  // ROS_INFO("Residual between point and line %lf", Rpos*pointToLineDist(gripperPos.back(), PCofObject));
  // ROS_INFO("Reward is %lf", reward);
  // ROS_INFO("\n\n");

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
  // result.push_back(vec.back());
  // typename std::vector<T>::const_iterator iter = --vec.end();
  // for(; iter!=vec.begin(); iter--){
  //   result.push_back(result.back()+*iter);
  // }
  // std::reverse(result.begin(),result.end());
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


bool DemoLearnManifold::doGraspAndLift() {

  addNoise();

  hiqp_msgs::Task gripperBelowUpperPlane;
  hiqp_msgs::Task gripperAboveLowerPlane;
  hiqp_msgs::Task gripperTomanifold;
  hiqp_msgs::Task gripperToGraspPlane;
  hiqp_msgs::Task gripperAxisToTargetAxis;
  hiqp_msgs::Task gripperAxisAlignedToTargetAxis;
  hiqp_msgs::Task jointLimitTasks1;
  hiqp_msgs::Task jointLimitTasks2;
  hiqp_msgs::Task jointLimitTasks3;
  hiqp_msgs::Task jointLimitTasks4;
  hiqp_msgs::Task jointLimitTasks5;
  hiqp_msgs::Task jointLimitTasks6;
  hiqp_msgs::Task jointLimitTasks7;

  hiqp_msgs::Primitive eef_point;
  hiqp_msgs::Primitive final_point;
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

  final_point = hiqp_ros::createPrimitiveMsg(
    "final_point", "point", "world", true, {0, 0, 1, 1},
    {finalPos[0], finalPos[1], finalPos[2]});

  manifold = hiqp_ros::createPrimitiveMsg(
    "grasp_manifold", "cylinder", "world", true, {1.0, 0.0, 0.0, 0.3},
    {0, 0, 1,manifoldPos[0], manifoldPos[1], manifoldPos[2],
      manifold_radius_,manifold_height_});

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

  // gripperTomanifold = hiqp_ros::createTaskMsg(
  //   "gripper_ee_point_on_grasp_cylinder", 2, true, false, true,
  //   {"TDefGeomProj", "point", "cylinder",
  //   eef_point.name + " = " + manifold.name},
  //   {"TDynLinear", std::to_string(decay_rate_ * DYNAMICS_GAIN)});

  gripperAxisAlignedToTargetAxis = hiqp_ros::createTaskMsg(
    "gripper_vertical_axis_parallel_grasp_target_axis", 2, true, false, true,
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

  hiqp_client_.setPrimitives({eef_point, gripper_approach_axis, gripper_vertical_axis, grasp_target_axis, grasp_plane, manifold, final_point, PC_of_object});

  hiqp_client_.setTasks({gripperToGraspPlane, gripperAxisToTargetAxis, gripperTomanifold, gripperAxisAlignedToTargetAxis});

  hiqp_client_.activateTasks({gripperToGraspPlane.name, gripperAxisToTargetAxis.name,
    gripperTomanifold.name, gripperAxisAlignedToTargetAxis.name});

  hiqp_client_.waitForCompletion(
    {gripperToGraspPlane.name, gripperAxisToTargetAxis.name,
      gripperTomanifold.name, gripperAxisAlignedToTargetAxis.name},
      {TaskDoneReaction::REMOVE, TaskDoneReaction::REMOVE, TaskDoneReaction::REMOVE, TaskDoneReaction::REMOVE},
      {1e-10,1e-10,1e-10,1e-10}, exec_time_);

  hiqp_client_.removeTasks({gripperToGraspPlane.name, gripperAxisToTargetAxis.name,
    gripperTomanifold.name, gripperAxisAlignedToTargetAxis.name});


  // Remove all primitives which are the same for all tasks 

  hiqp_client_.removePrimitives({eef_point.name, gripper_approach_axis.name, gripper_vertical_axis.name,
    grasp_target_axis.name, grasp_plane.name, final_point.name, manifold.name, PC_of_object.name});


  finish_recording_.publish(finish_msg_);


  // --------------- //
  // --- Extract --- //
  // --------------- //


  return true;
}

bool DemoLearnManifold::startDemo(std_srvs::Empty::Request& req,
 std_srvs::Empty::Response& res) {
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

  if (!doGraspAndLift()) {
    ROS_ERROR("Could not set the grasp approach!");
    safeShutdown();
    return false;
  }


  ROS_INFO("Updating policy");
  if (!init){
    updatePolicy();
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
