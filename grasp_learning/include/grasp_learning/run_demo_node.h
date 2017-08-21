#ifndef RUN_DEMO_NODE_H
#define RUN_DEMO_NODE_H

#include <rosbag/bag.h>
#include "std_msgs/Bool.h"
#include <std_msgs/Empty.h>
#include <std_srvs/Empty.h>
#include "sensor_msgs/JointState.h"
#include "ros/ros.h"
#include <grasp_learning/FinishRecording.h>
#include <vector>

namespace hiqp{
	namespace grasp_learner{
		class runDemoNode
		{
		public:
			runDemoNode(){
				nh_ = ros::NodeHandle("~");
				start_demo_client_ = nh.serviceClient<std_srvs::Empty>("demo_learn_manifold/start_demo");
				reset_policy_search_node_client_ = nh.serviceClient<std_srvs::Empty>("reset_node");
    			nh_.param<int>("max_rollouts", max_rollouts_, 100);
				current_rollout_ = 0;
			};
			~runDemoNode(){};
		
			template <typename ROSMessageType>
  			int addSubscription(ros::NodeHandle& controller_nh, const std::string& topic_name, unsigned int buffer_size){
  							ros::Subscriber sub;
  				    sub = controller_nh.subscribe(topic_name, buffer_size,	&runDemoNode::topicCallback<ROSMessageType>, this);
    				ROS_INFO_STREAM("Subsribed to topic '" << topic_name << "'");
    				subs_.push_back(sub);
  			}


			template <typename ROSMessageType>
  			void topicCallback(const ROSMessageType& msg);
		

		private:
			std::vector<ros::Subscriber> subs_;
			ros::NodeHandle nh;
  			ros::NodeHandle nh_;
			ros::ServiceClient start_demo_client_;
			ros::ServiceClient reset_policy_search_node_client_;

			int max_rollouts_;
			int current_rollout_;
			std_srvs::Empty start_demo_srv_;
			std_srvs::Empty reset_policy_search_node_srv_;
		};
	}
}

#endif  // Include guard
