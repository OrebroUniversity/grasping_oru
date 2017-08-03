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
    nh_.param<double>("manifold_pos_x_", manifold_pos_x_, 0);
    nh_.param<double>("manifold_pos_y_", manifold_pos_y_, 0);
    nh_.param<double>("manifold_pos_z_", manifold_pos_z_, 0);

    nh_.param<int>("burn_in_trials",  burn_in_trials_, 8);
    nh_.param<int>("max_num_samples", max_num_samples_, 8);

    nh_.param<std::string>("task", task_name_, "gripperToHorizontalPlane");

    if (with_gazebo_) ROS_INFO("Grasping experiments running in Gazebo.");

  // register general callbacks
    start_demo_srv_ =
    nh_.advertiseService("start_demo", &DemoLearnManifold::startDemo, this);

    set_gazebo_physics_clt_ = n_.serviceClient<gazebo_msgs::SetPhysicsProperties>(
      "/gazebo/set_physics_properties");

    client_Policy_Search_ = n_.serviceClient<grasp_learning::PolicySearch>("policy_Search");
    client_Add_Noise_ = n_.serviceClient<grasp_learning::AddNoise>("add_Noise");

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

  ROS_INFO("DEMO LEARNING READY.");

}

void DemoLearnManifold::safeShutdown() {
  hiqp_client_.resetHiQPController();
  ROS_BREAK();  // I must break you ... ros::shutdown() doesn't seem to do the
                // job
}

void DemoLearnManifold::safeReset() { hiqp_client_.resetHiQPController(); }


void DemoLearnManifold::updatePolicy(){

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

void DemoLearnManifold::addNoise(){

  grasp_learning::AddNoise srv_;

  srv_.request.str = " ";  


  ROS_INFO("Adding Noise to Parameter Vector");
  if (client_Add_Noise_.call(srv_)){
    ROS_INFO("Added noise successfully");
  }
  else{
   ROS_INFO("Could not add nosie"); 
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

  hiqp_msgs::Task gripperBelowUpperPlane;
  hiqp_msgs::Task gripperAboveLowerPlane;
  hiqp_msgs::Task gripperTomanifold;

  hiqp_msgs::Primitive eef_point;
  hiqp_msgs::Primitive manifold;

  // Define the primitives
  eef_point = hiqp_ros::createPrimitiveMsg(
    "point_eef", "point", grasp_lower_.e_frame_, true, {1, 0, 0, 1},
    {grasp_lower_.e_(0), grasp_lower_.e_(1), grasp_lower_.e_(2)+0.1});


  manifold.name = "grasp_manifold";
  manifold.type = "cylinder";
    manifold.frame_id = "world";  // TODO: Check this.
    manifold.visible = true;
    manifold.color = {1.0, 0.0, 0.0, 0.3};
    manifold.parameters = {
        0, 0, 1,  // Axis first.
        manifold_pos_x_, manifold_pos_y_, manifold_pos_z_,  // Position x
        manifold_radius_,  // radius
        manifold_height_};  // height - shouldn't really matter unless the
                             // gripper is "YUGE".

        grasp_lower_.plane = hiqp_ros::createPrimitiveMsg(
          "lower_plane", "plane", "world", true, {0, 0, 1, 0.2},
    {0, 0, 1,1});//0.89

        grasp_upper_.plane = hiqp_ros::createPrimitiveMsg(
          "upper_plane", "plane", "world", true, {0, 0, 1, 0.2},
    {0, 0, 1,1.1});//0.89


  // Define tasks

        gripperBelowUpperPlane = hiqp_ros::createTaskMsg(
          "point_below_upper_plane", 1, true, false, true,
          {"TDefGeomProj", "point", "plane",
          eef_point.name + " < " + grasp_upper_.plane.name},
          {"TDynLinear", std::to_string(decay_rate_ * DYNAMICS_GAIN)});

        gripperAboveLowerPlane = hiqp_ros::createTaskMsg(
          "point_above_lower_plane", 1, true, false, true,
          {"TDefGeomProj", "point", "plane",
          eef_point.name + " > " + grasp_lower_.plane.name},
          {"TDynLinear", std::to_string(decay_rate_ * DYNAMICS_GAIN)});

        gripperTomanifold = hiqp_ros::createTaskMsg(
          "point_to_manifold", 1, true, false, true,
          {"TDefGeomProj", "point", "cylinder",
          eef_point.name + " = " + manifold.name},
          {"TDynRBFN", std::to_string(decay_rate_ * DYNAMICS_GAIN), std::to_string(num_kernels_),
          std::to_string(num_kernel_rows_), std::to_string(manifold_radius_), std::to_string(manifold_height_),
          std::to_string(manifold_pos_x_), std::to_string(manifold_pos_y_),std::to_string(manifold_pos_z_),
          std::to_string(burn_in_trials_), std::to_string(max_num_samples_)}); 

        start_recording_.publish(start_msg_);

        using hiqp_ros::TaskDoneReaction;

        if(initialize){
          hiqp_client_.setPrimitives({eef_point, grasp_lower_.plane, grasp_upper_.plane, manifold});
          hiqp_client_.setTasks({gripperBelowUpperPlane, gripperAboveLowerPlane, gripperTomanifold});
          initialize = false;
        }

         // addNoise(); // Call the service that adds noise to the parameter vector
         // hiqp_client_.activateTasks({gripperBelowUpperPlane.name, gripperAboveLowerPlane.name, gripperTomanifold.name});
         hiqp_client_.waitForCompletion(
          {gripperBelowUpperPlane.name, gripperAboveLowerPlane.name, gripperTomanifold.name},
          {TaskDoneReaction::REMOVE, TaskDoneReaction::REMOVE, TaskDoneReaction::NONE},
          {1e-10,1e-10,1e-10}, exec_time_);

         // hiqp_client_.removePrimitives({eef_point.name, grasp_lower_.plane.name, grasp_upper_.plane.name, manifold.name});

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
      hiqp_client_.setJointAngles(sensing_config_);


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
