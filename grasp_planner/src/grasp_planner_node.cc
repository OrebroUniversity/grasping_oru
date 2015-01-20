#include <grasp_planner/grasp_planner_node.hh>

using namespace std;


int main(int argc, char **argv) {

    ros::init(argc,argv,"grasp_planner_node");
    
    GraspPlannerNode nd;

    ros::spin();
    return 0;
}
