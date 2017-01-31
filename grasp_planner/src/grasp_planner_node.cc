#include <grasp_planner/grasp_planner.h>

int main(int argc, char **argv) {

  using grasp_planner::GraspPlanner;
  
  ros::init(argc,argv,"grasp_planner_node");
  
  SDF_Parameters param;
  
  //Pose Offset as a transformation matrix
  Eigen::Matrix4d initialTransformation = 
    Eigen::MatrixXd::Identity(4,4);
  
  param.pose_offset = initialTransformation;
  GraspPlanner gp(param);
  ros::AsyncSpinner spinner(4); // Use 4 threads
  spinner.start();
  gp.publishPC();
  ros::waitForShutdown();
  
  return 0;
}
