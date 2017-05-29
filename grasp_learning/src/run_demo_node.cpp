#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <string>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
#include <grasp_learning/run_demo_node.h>
namespace hiqp{
  namespace grasp_learner{



    template <>
    void runDemoNode::topicCallback<std_msgs::Empty>(const std_msgs::Empty& msg){
      current_rollout_++;
      if (current_rollout_<max_rollouts_){
        ROS_INFO("Rollout number %d", current_rollout_);
        start_demo_client_.call(start_demo_srv_);
      }
      else if (current_rollout_==max_rollouts_)
      {
        current_rollout_=0;
        reset_policy_search_node_client_.call(reset_policy_search_node_srv_);
      }
    }
  }
}


int main(int argc, char **argv) {

  using hiqp::grasp_learner::runDemoNode;
  
  ros::init(argc,argv,"run_demo_node");
  
  ros::NodeHandle nh;
  runDemoNode runDemoNode_;

  runDemoNode_.addSubscription<std_msgs::Empty>(nh, "/run_new_episode",1);

  ros::AsyncSpinner spinner(1); // Use 4 threads
  spinner.start();
  ros::waitForShutdown();
  
  return 0;
}
