#include <grasp_planner/grasp_planner_node.hh>

using namespace std;


int main(int argc, char **argv) {

    ros::init(argc,argv,"grasp_planner_node");
    
    SDF_Parameters param;

     //Pose Offset as a transformation matrix
    Eigen::Matrix4d initialTransformation = 
    Eigen::MatrixXd::Identity(4,4);

    //define translation offsets in x y z
    initialTransformation(0,3) = 0.0;  //x 
    initialTransformation(1,3) = 0.0;  //y
    initialTransformation(2,3) = -0.7; //z

    param.pose_offset = initialTransformation;
    GraspPlannerNode nd(param);

    ros::spin();
    return 0;
}
