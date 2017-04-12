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
				max_num_exec_ = 100;
				start_demo_client_ = nh.serviceClient<std_srvs::Empty>("demo_learning/start_demo");
				current_exec_ = 0;
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
			ros::ServiceClient start_demo_client_;
			unsigned int max_num_exec_;
			unsigned int current_exec_;
			std_srvs::Empty start_demo_srv_;
		};
	}
}

#endif  // Include guard