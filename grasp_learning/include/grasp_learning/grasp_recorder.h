#ifndef GRASP_RECORDER_H
#define GRASP_RECORDER_H

#include <rosbag/bag.h>
#include "std_msgs/Bool.h"
#include <std_msgs/Empty.h>
#include "sensor_msgs/JointState.h"
#include "ros/ros.h"
#include <hiqp_msgs/TaskMeasures.h>
#include <grasp_learning/StartRecording.h>
#include <grasp_learning/FinishRecording.h>
#include <vector>
namespace hiqp{
	namespace grasp_learner{
		class grasp_recorder
		{
		public:
			grasp_recorder(std::string file_name){
				file_name_ = file_name;
				num_record_ = 0;};
			~grasp_recorder(){};
		
			template <typename ROSMessageType>
  			int addSubscription(ros::NodeHandle& controller_nh, const std::string& topic_name, unsigned int buffer_size){
  							ros::Subscriber sub;
  				    sub = controller_nh.subscribe(topic_name, buffer_size,	&grasp_recorder::topicCallback<ROSMessageType>, this);
    				ROS_INFO_STREAM("Subsribed to topic '" << topic_name << "'");
    				subs_.push_back(sub);
  			}


			template <typename ROSMessageType>
  			void topicCallback(const ROSMessageType& msg);
		

		private:
			std::vector<ros::Subscriber> subs_;

			std::vector< sensor_msgs::JointState > joint_state_vec_;
			std::vector< hiqp_msgs::TaskMeasures > task_dynamics_vec_;
			std::string file_name_;
			int num_record_;
			bool record_ = true;
		};
	}
}

#endif  // Include guard