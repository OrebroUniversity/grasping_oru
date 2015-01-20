#ifndef __GRASP_PLANNER_NODE_HH
#define __GRASP_PLANNER_NODE_HH

#include <constraint_map/SimpleOccMap.hh>
#include <constraint_map/ConstraintMap.hh>
#include <constraint_map/SimpleOccMapMsg.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <ros/ros.h>

class GraspPlannerNode {

    private:
	// Our NodeHandle, points to home
	ros::NodeHandle nh_;
	//global node handle
	ros::NodeHandle n_;

	ros::Publisher gripper_map_publisher_;
	ros::Publisher object_map_publisher_;

	std::string gripper_fname;
	std::string gripper_frame_name;
	std::string object_map_frame_name;
	std::string gripper_map_topic;
	std::string object_map_topic;

    public:
	tf::TransformBroadcaster br;
	ConstraintMap *gripper_map;
	
	GraspPlannerNode() {

	    nh_ = ros::NodeHandle("~");
	    n_ = ros::NodeHandle();

	    nh_.param<std::string>("gripper_file",gripper_fname,"full.cons");
	    nh_.param<std::string>("gripper_frame_name",gripper_frame_name,"gripper_frame");
	    nh_.param<std::string>("map_frame_name",object_map_frame_name,"map_frame");
	    nh_.param<std::string>("map_topic",object_map_topic,"object_map");
	    nh_.param<std::string>("gripper_map_topic",gripper_map_topic,"gripper_map");
	    
	    
	    gripper_map_publisher_ = nh_.advertise<constraint_map::SimpleOccMapMsg> (gripper_map_topic,10);
	    object_map_publisher_ = nh_.advertise<constraint_map::SimpleOccMapMsg> (object_map_topic,10);
	    
	    gripper_map = new ConstraintMap();
	    bool success = gripper_map->loadGripperConstraints(gripper_fname.c_str());
	
	    if(!success) {
		    ROS_ERROR("could not load gripper constraints file from %s",gripper_fname.c_str());
		    ros::shutdown();
	    }
	}
	~GraspPlannerNode() {
	    delete gripper_map;
	}

	void publishMap() {
	    /*
	    constraint_map::SimpleOccMapMsg msg;
	    object_map->toMessage(msg);
	    msg.header.frame_id = "/map_frame";
	    object_map_publisher_.publish(msg); */
	    
	    ROS_INFO("Publishing map");
	    constraint_map::SimpleOccMapMsg msg2;
	    gripper_map->toMessage(msg2);
	    msg2.header.frame_id = gripper_frame_name;
	    gripper_map_publisher_.publish(msg2);

	}


};

#endif
