#pragma once

#include <constraint_map/simple_occ_map.h>
#include <constraint_map/constraint_map.h>
#include <constraint_map/SimpleOccMapMsg.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <eigen_conversions/eigen_msg.h>
#include <std_srvs/Empty.h>

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <sdf_tracker/sdf_tracker.h>

#include <boost/thread/mutex.hpp>
#include <grasp_planner/PlanGrasp.h>
#include <grasp_planner/LoadResource.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <sdf_tracker_msgs/SDFMap.h>

#define POINT_SCALE  0.02
#define LINE_SCALE   0.3
#define PLANE_SCALE  1
#define CONE_SCALE   0.3
#define LINE_WIDTH   0.005

namespace grasp_planner {
/** This class implements the grasp planner node.
    It contains:
    * a constraint map, loaded from file
    * a TSDF map constructed online or loaded from file
    * functionalities to compute constraint envelopes

TODO: Here update service calls to return constraints in terms of hiqp controller messages
*/
class GraspPlanner {

private:
	// Our NodeHandle, points to home
	ros::NodeHandle nh_;
	//global node handle
	ros::NodeHandle n_;
	boost::mutex tracker_m;
	SDFTracker* myTracker_;
	SDF_Parameters myParameters_;
	SDF_CamParameters cam1params_, cam2params_;
	ConstraintMap *gripper_map;
	Eigen::Affine3d cam2map, prev_cam2map;
	
	ros::Publisher sdf_map_publisher_;
	ros::Publisher fused_pc_publisher_;
	ros::Publisher vis_pub_;
	ros::Publisher constraint_pub_;
	ros::Subscriber depth_subscriber_;
	ros::Subscriber depth_camera_info_subscriber_;
	ros::Subscriber depth_subscriber2_;
	ros::Subscriber depth_camera_info_subscriber2_;
	ros::Timer heartbeat_tf_;
	ros::Timer heartbeat_map_;

	ros::ServiceServer plan_grasp_serrver_;
	ros::ServiceServer publish_map_server_;
	ros::ServiceServer save_map_server_;
	ros::ServiceServer clear_map_server_;
	ros::ServiceServer load_volume_server_;
	ros::ServiceServer load_constraints_server_;
	ros::ServiceServer map_to_edt_;

	tf::TransformBroadcaster br;
	tf::TransformListener tl;
	tf::Transform gripper2map;

	std::string gripper_fname;
	std::string gripper_frame_name;
	std::string object_map_frame_name;
	std::string gripper_map_topic;
	std::string sdf_map_topic;
	std::string object_map_topic;
	std::string depth_topic_name_;
	std::string depth_info_topic_name_;
	std::string depth_topic_name2_;
	std::string depth_info_topic_name2_;
	std::string camera_link_;
	std::string fused_pc_topic;
	std::string loadVolume_;
	std::string dumpfile;

	int skip_frames_, frame_counter_;
	bool use_tf_, grasp_frame_set, publish_pc, isInfoSet1, isInfoSet2;
	double cylinder_tolerance, plane_tolerance;//, orientation_tolerance;
	int MIN_ENVELOPE_VOLUME;

public:
  GraspPlanner(SDF_Parameters& parameters);

  ~GraspPlanner();
	
  void publishMap(const ros::TimerEvent& event);

  void publishTF(const ros::TimerEvent& event);

  void publishPC();

  void depthInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg);
	
	void depthInfoCallback2(const sensor_msgs::CameraInfo::ConstPtr& msg);

  void depthCallback(const sensor_msgs::Image::ConstPtr& msg);

  void depthCallback2(const sensor_msgs::Image::ConstPtr& msg);
	
	bool loadConstraintsCallback(grasp_planner::LoadResource::Request  &req,
                                 grasp_planner::LoadResource::Response &res ); 

	bool loadVolumeCallback(grasp_planner::LoadResource::Request  &req,
                          grasp_planner::LoadResource::Response &res );
  	
	bool mapToEdtCallback(std_srvs::Empty::Request  &req,
                           std_srvs::Empty::Response &res );
	
	bool saveMapCallback(std_srvs::Empty::Request  &req,
                       std_srvs::Empty::Response &res );
	
	bool clearMapCallback(std_srvs::Empty::Request  &req,
                        std_srvs::Empty::Response &res );
	
	bool publishMapCallback(std_srvs::Empty::Request  &req,
                          std_srvs::Empty::Response &res );

	bool planGraspCallback(grasp_planner::PlanGrasp::Request  &req,
                         grasp_planner::PlanGrasp::Response &res );

	void addPlaneMarker(visualization_msgs::MarkerArray& markers,
                      Eigen::Vector3d n,
                      double d,
                      std::string frame_, 
                      std::string namespc="plane",
                      double r=1,
                      double g=0.2,
                      double b=0.5);
  
  void addCylinderMarker(visualization_msgs::MarkerArray& markers,
                         Eigen::Vector3d p,
                         Eigen::Vector3d v,
                         double r,
                         std::string frame_,
                         double h=LINE_SCALE,
                         std::string namespc="cylinder",
                         double rc=0,
                         double g=1,
                         double b=1);

  void addSphereMarker(visualization_msgs::MarkerArray& markers,
                       Eigen::Vector3f center,
                       float radius,
                       std::string frame_,
                       std::string namespc="sphere",
                       double r=0,
                       double g=1,
                       double b=1);

};
  
}
