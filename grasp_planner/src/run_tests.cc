#include "ros/ros.h"
#include <grasp_planner/PlanGrasp.h>
#include <grasp_planner/LoadResource.h>
#include <std_srvs/Empty.h>
#include <fstream>
#include <iostream>
#include <cstdlib>

#define TEDDY_R 0.085
#define TEDDY_H 0.0
#define BOTTLE_R 0.06
#define BOTTLE_H 0.2
#define CUP_R 0.08
#define CUP_H 0.1
#define BOX_R 0.07
#define BOX_H 0.15

int main(int argc, char **argv)
{
    ros::init(argc, argv, "run_tests");

    bool do_debug = false;

    ros::NodeHandle n;
    ros::ServiceClient empty_client = n.serviceClient<std_srvs::Empty>("/gplanner/publish_map");
    ros::ServiceClient volume_client = n.serviceClient<grasp_planner::LoadResource>("/gplanner/load_volume");
    ros::ServiceClient constraint_client = n.serviceClient<grasp_planner::LoadResource>("/gplanner/load_constraints");
    ros::ServiceClient plan_grasp_client = n.serviceClient<grasp_planner::PlanGrasp>("/gplanner/plan_grasp");
    
    grasp_planner::LoadResource volume_path;
    grasp_planner::LoadResource constraint_path;
    grasp_planner::PlanGrasp plan;
    std_srvs::Empty empty;

    std::string path_prefix = "/home/tsv/code/workspace-kuka/src/grasping_oru/grasp_planner/cfg/";
    plan.request.header.stamp = ros::Time::now();
    plan.request.header.frame_id = "roi_frame";
    plan.request.objectPose.orientation.w = 1;

    std::ifstream ifs ("test_10mm.txt", std::ifstream::in);
    std::ofstream ofs ("res_10mm.m", std::ofstream::out);
    int i=1;
    while(ifs.good() && !ifs.eof()) {
	std::string vol_str;
	ifs>>vol_str;
	if(vol_str=="") break;
	std::cout<<"processing "<<vol_str<<std::endl;
	ofs<<"names{"<<i<<"} = \'"<<vol_str<<"\';\n";
	float x,y,z;
	std::vector<float> times,volumes;
	/////////////10mm///////////////
	//first volume
	volume_path.request.name = path_prefix+vol_str;
	if (!volume_client.call(volume_path))
	{
	    ROS_ERROR("Failed to call service load_volume");
	    return 1;
	}
	if(do_debug) {
	    empty_client.call(empty);
	}

	//teddy and pig
	constraint_path.request.name = path_prefix+"sphere_10mm.cons";
	if (!constraint_client.call(constraint_path))
	{
	    ROS_ERROR("Failed to call service load_constraints");
	    return 1;
	}
	ifs>>x>>y>>z;
	plan.request.objectPose.position.x = x;
	plan.request.objectPose.position.y = y;
	plan.request.objectPose.position.z = z;
	plan.request.object_radius = TEDDY_R;
	plan.request.object_height = TEDDY_H;
	if (!plan_grasp_client.call(plan))
	{
	    ROS_ERROR("Failed to call service load_constraints");
	    return 1;
	}
	std::cout<<"volume: "<<plan.response.volume<<" time: "<<plan.response.time<<std::endl;
	times.push_back(plan.response.time);
	volumes.push_back(plan.response.volume);

	ifs>>x>>y>>z;
	plan.request.objectPose.position.x = x;
	plan.request.objectPose.position.y = y;
	plan.request.objectPose.position.z = z;
	plan.request.object_radius = TEDDY_R;
	plan.request.object_height = TEDDY_H;
	if (!plan_grasp_client.call(plan))
	{
	    ROS_ERROR("Failed to call service load_constraints");
	    return 1;
	}
	std::cout<<"volume: "<<plan.response.volume<<" time: "<<plan.response.time<<std::endl;
	times.push_back(plan.response.time);
	volumes.push_back(plan.response.volume);

	//cup, bottle, box
	constraint_path.request.name = path_prefix+"cylinder_10mm.cons";
	if (!constraint_client.call(constraint_path))
	{
	    ROS_ERROR("Failed to call service load_constraints");
	    return 1;
	}
	ifs>>x>>y>>z;
	plan.request.objectPose.position.x = x;
	plan.request.objectPose.position.y = y;
	plan.request.objectPose.position.z = z;
	plan.request.object_radius = BOTTLE_R;
	plan.request.object_height = BOTTLE_H;
	if (!plan_grasp_client.call(plan))
	{
	    ROS_ERROR("Failed to call service load_constraints");
	    return 1;
	}
	std::cout<<"volume: "<<plan.response.volume<<" time: "<<plan.response.time<<std::endl;
	times.push_back(plan.response.time);
	volumes.push_back(plan.response.volume);

	ifs>>x>>y>>z;
	plan.request.objectPose.position.x = x;
	plan.request.objectPose.position.y = y;
	plan.request.objectPose.position.z = z;
	plan.request.object_radius = CUP_R;
	plan.request.object_height = CUP_H;
	if (!plan_grasp_client.call(plan))
	{
	    ROS_ERROR("Failed to call service load_constraints");
	    return 1;
	}
	std::cout<<"volume: "<<plan.response.volume<<" time: "<<plan.response.time<<std::endl;
	times.push_back(plan.response.time);
	volumes.push_back(plan.response.volume);

	ifs>>x>>y>>z;
	plan.request.objectPose.position.x = x;
	plan.request.objectPose.position.y = y;
	plan.request.objectPose.position.z = z;
	plan.request.object_radius = BOX_R;
	plan.request.object_height = BOX_H;
	if (!plan_grasp_client.call(plan))
	{
	    ROS_ERROR("Failed to call service load_constraints");
	    return 1;
	}
	std::cout<<"volume: "<<plan.response.volume<<" time: "<<plan.response.time<<std::endl;
	times.push_back(plan.response.time);
	volumes.push_back(plan.response.volume);
	
	ofs<<"times("<<i<<",:) = [";
	for(int q=0; q<times.size(); q++) {
	    ofs<<times[q]<<" ";
	}
	ofs<<"];\n";
	ofs<<"volumes("<<i<<",:) = [";
	for(int q=0; q<volumes.size(); q++) {
	    ofs<<volumes[q]<<" ";
	}
	ofs<<"];\n";

	i++;
    }

    return 0;
}
