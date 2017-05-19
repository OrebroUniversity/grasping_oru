#include <grasp_learning/grasp_recorder.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <string>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
namespace hiqp{
	namespace grasp_learner{


		template <>
		void grasp_recorder::topicCallback<sensor_msgs::JointState>(const sensor_msgs::JointState& msg){
			// std::cout<<"Startoring joint states"<<std::endl;
			if (record_)
				this->joint_state_vec_.push_back(msg);
		}

		template <>
		void grasp_recorder::topicCallback<hiqp_msgs::TaskMeasures>(const hiqp_msgs::TaskMeasures& msg){
			// std::cout<<"Storing task dynamics"<<std::endl;
			if (record_)
				this->task_dynamics_vec_.push_back(msg);
		}

		template <>
		void grasp_recorder::topicCallback<grasp_learning::StartRecording>(const grasp_learning::StartRecording& msg){
			std::cout<<"Start recording, empty previous vectors"<<std::endl;
			this->task_dynamics_vec_.clear();
			this->joint_state_vec_.clear();
			record_ = true;

		}


		template <>
		void grasp_recorder::topicCallback<grasp_learning::FinishRecording>(const grasp_learning::FinishRecording& msg){
			record_ = false;
			std::cout<<"One grasping episode finished"<<std::endl;
			// std::cout<<"Number of joint messages recorded "<<joint_state_vec_.size()<<std::endl;
			std::cout<<"Number of task messages recorded "<<task_dynamics_vec_.size()<<std::endl;
			std::ostringstream convert;   // stream used for the conversion

			convert << ++num_record_;      // insert the textual representation of 'Number' in the characters in the stream

			std::string task_dynamics_file_name = "../grasping_ws/src/grasping_oru/grasp_learning/stored_data/training_data/task_dynamics/"+ this->file_name_ + convert.str() +".txt";

			std::ofstream output_file2 (task_dynamics_file_name.c_str());
			if (output_file2.is_open()){
				for (float i=0;i<task_dynamics_vec_.size();i++){
					if (i==0){
						for(unsigned int j=0;j<task_dynamics_vec_[i].task_measures.size();j++){
							output_file2<<"task_performance "<<task_dynamics_vec_[i].task_measures[j].task_name<<" ";
						}
						output_file2 << "\n";
					}
					for(unsigned int j=0;j<task_dynamics_vec_[i].task_measures.size();j++){
						for(unsigned int z=0;z<task_dynamics_vec_[i].task_measures[j].de.size();z++){
							output_file2 << this->task_dynamics_vec_[i].task_measures[j].e[z]<<" ";
							output_file2 << this->task_dynamics_vec_[i].task_measures[j].de[z]<<" ";	
						}
					}
					output_file2 << "\n";
				}
			}
			output_file2.close();	

			std::cout<<"Finished storing data"<<std::endl;
		}



	}
}