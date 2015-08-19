#include <grasp_planner/grasp_planner_node.hh>

using namespace std;


int main(int argc, char **argv) {

    ros::init(argc,argv,"grasp_planner_node");
    
    SDF_Parameters param;

    //Pose Offset as a transformation matrix
    Eigen::Matrix4d initialTransformation = 
    Eigen::MatrixXd::Identity(4,4);

    param.pose_offset = initialTransformation;
    GraspPlannerNode nd(param);
    ros::AsyncSpinner spinner(4); // Use 4 threads
    spinner.start();
    nd.publishPC();
    ros::waitForShutdown();

    return 0;
}
