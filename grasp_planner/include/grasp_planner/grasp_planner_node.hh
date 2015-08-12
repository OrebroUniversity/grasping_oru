#ifndef __GRASP_PLANNER_NODE_HH
#define __GRASP_PLANNER_NODE_HH

#include <constraint_map/SimpleOccMap.hh>
#include <constraint_map/ConstraintMap.hh>
#include <constraint_map/SimpleOccMapMsg.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <eigen_conversions/eigen_msg.h>
#include <std_srvs/Empty.h>

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <sdf_tracker/sdf_tracker.h>

#include <boost/thread/mutex.hpp>
#include <grasp_planner/PlanGrasp.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

class GraspPlannerNode {

    private:
	// Our NodeHandle, points to home
	ros::NodeHandle nh_;
	//global node handle
	ros::NodeHandle n_;
	boost::mutex tracker_m;
	SDFTracker* myTracker_;
	SDF_Parameters myParameters_;
	ConstraintMap *gripper_map;
	
	ros::Publisher gripper_map_publisher_;
	ros::Publisher object_map_publisher_;
	ros::Publisher fused_pc_publisher_;
	ros::Subscriber depth_subscriber_;
	ros::Timer heartbeat_tf_;
	ros::Timer heartbeat_pc_;

	ros::ServiceServer publish_maps_;
	ros::ServiceServer plan_grasp_serrver_;

	tf::TransformBroadcaster br;
	tf::TransformListener tl;
	tf::Transform gripper2map;

	std::string gripper_fname;
	std::string gripper_frame_name;
	std::string object_map_frame_name;
	std::string gripper_map_topic;
	std::string object_map_topic;
	std::string depth_topic_name_;
	std::string camera_link_;
	std::string fused_pc_topic;
	std::string loadVolume_;

	int skip_frames_, frame_counter_;
	bool use_tf_, grasp_frame_set;

	//void depthCallback(const sensor_msgs::Image::ConstPtr& msg);
    public:
	GraspPlannerNode(SDF_Parameters &parameters) {

	    nh_ = ros::NodeHandle("~");
	    n_ = ros::NodeHandle();

	    nh_.param<std::string>("gripper_file",gripper_fname,"full.cons");
	    nh_.param<std::string>("grasp_frame_name",gripper_frame_name,"planned_grasp");
	    nh_.param<std::string>("map_frame_name",object_map_frame_name,"map_frame");
	    nh_.param<std::string>("map_topic",object_map_topic,"object_map");
	    nh_.param<std::string>("gripper_map_topic",gripper_map_topic,"gripper_map");
	    nh_.param<std::string>("fused_pc_topic",fused_pc_topic,"fused_pc");
	    nh_.param<std::string>("LoadVolume", loadVolume_,"none");

	    gripper_map = new ConstraintMap();
	    bool success = gripper_map->loadGripperConstraints(gripper_fname.c_str());
	
	    if(!success) {
		    ROS_ERROR("could not load gripper constraints file from %s",gripper_fname.c_str());
		    ros::shutdown();
	    }
	   
	    //Parameters for SDF tracking 
	    myParameters_ = parameters;
	    //node specific parameters
	    nh_.param("use_tf", use_tf_, false);
	    nh_.param<std::string>("depth_topic_name", depth_topic_name_,"/camera/depth/image");
	    nh_.param<std::string>("camera_link", camera_link_,"/camera/depth_frame");
	    nh_.param("skip_frames", skip_frames_, 0);

	    //parameters used for the SDF tracker
	    nh_.getParam("ImageWidth", myParameters_.image_width);
	    nh_.getParam("ImageHeight", myParameters_.image_height); 
	    nh_.getParam("InteractiveMode", myParameters_.interactive_mode);
	    nh_.getParam("MaxWeight",myParameters_.Wmax);
	    //nh_.getParam("CellSize",myParameters_.resolution);
	    nh_.getParam("GridSizeX",myParameters_.XSize);
	    nh_.getParam("GridSizeY",myParameters_.YSize);
	    nh_.getParam("GridSizeZ",myParameters_.ZSize);
	    nh_.getParam("PositiveTruncationDistance",myParameters_.Dmax);
	    nh_.getParam("NegativeTruncationDistance",myParameters_.Dmin);
	    nh_.getParam("RobustStatisticCoefficient", myParameters_.robust_statistic_coefficient);
	    nh_.getParam("Regularization", myParameters_.regularization);
	    nh_.getParam("MinPoseChangeToFuseData", myParameters_.min_pose_change);
	    nh_.getParam("ConvergenceCondition", myParameters_.min_parameter_update);
	    nh_.getParam("MaximumRaycastSteps", myParameters_.raycast_steps);
	    nh_.getParam("FocalLengthX", myParameters_.fx);
	    nh_.getParam("FocalLengthY", myParameters_.fy);
	    nh_.getParam("CenterPointX", myParameters_.cx);
	    nh_.getParam("CenterPointY", myParameters_.cy);


	    myParameters_.resolution = gripper_map->getResolution();
	    myTracker_ = new SDFTracker(myParameters_);

	    if(loadVolume_.compare(std::string("none"))!=0)
	    {
		myTracker_->LoadSDF(loadVolume_);
	    }
	    
	    //subscribe / advertise
	    gripper_map_publisher_ = nh_.advertise<constraint_map::SimpleOccMapMsg> (gripper_map_topic,10);
	    object_map_publisher_ = nh_.advertise<constraint_map::SimpleOccMapMsg> (object_map_topic,10);
	    fused_pc_publisher_ = nh_.advertise<sensor_msgs::PointCloud2> (fused_pc_topic,10);

	    depth_subscriber_ = n_.subscribe(depth_topic_name_, 1, &GraspPlannerNode::depthCallback, this);
	    publish_maps_ = nh_.advertiseService("publish_maps", &GraspPlannerNode::publish_map_callback, this);
	    plan_grasp_serrver_ = nh_.advertiseService("plan_grasp", &GraspPlannerNode::plan_grasp_callback, this);
	    
	    heartbeat_tf_ = nh_.createTimer(ros::Duration(0.1), &GraspPlannerNode::publishTF, this);
	    heartbeat_pc_ = nh_.createTimer(ros::Duration(10), &GraspPlannerNode::publishPC, this);

	    frame_counter_ = 0;
	    grasp_frame_set=false;
	}
	~GraspPlannerNode() {
	    if( gripper_map != NULL ) {
		delete gripper_map;
	    }
	    if(myTracker_ != NULL) 
	    {
		delete myTracker_;
	    }
	}

        void publishTF(const ros::TimerEvent& event) {
	    if(grasp_frame_set) {
		br.sendTransform(tf::StampedTransform(gripper2map, ros::Time::now(), object_map_frame_name, gripper_frame_name));
	    }
	}
        void publishPC(const ros::TimerEvent& event) {
	    ROS_INFO("Generating Triangles");
	    pcl::PointCloud<pcl::PointXYZ> pc;
	    pcl::PointXYZ pt;
	    tracker_m.lock();
	    myTracker_->triangles_.clear();
	    myTracker_->MakeTriangles();
	    for(int i=0; i<myTracker_->triangles_.size(); ++i) {
		pt.x = myTracker_->triangles_[i](0);
		pt.y = myTracker_->triangles_[i](1);
		pt.z = myTracker_->triangles_[i](2);
		pc.points.push_back(pt);
	    }
	    tracker_m.unlock();    
	    pc.is_dense = false;
	    pc.height = pc.points.size();
	    pc.width = 1;
	    ROS_INFO("Publishing PC");
	    sensor_msgs::PointCloud2 cloud;
	    pcl::toROSMsg(pc,cloud);
	    cloud.header.frame_id = object_map_frame_name;
	    cloud.header.stamp = ros::Time::now();
	    fused_pc_publisher_.publish(cloud);
	}

	void depthCallback(const sensor_msgs::Image::ConstPtr& msg)
	{
	    tf::StampedTransform camera_frame_to_map;
	    try {
		tl.waitForTransform(object_map_frame_name, camera_link_, ros::Time(0), ros::Duration(1.0) );
		tl.lookupTransform(object_map_frame_name,camera_link_, ros::Time(0), camera_frame_to_map);
	    } catch (tf::TransformException ex) {
		ROS_ERROR("%s",ex.what());
		return;
	    }
	    Eigen::Affine3d cam2map;
	    tf::transformTFToEigen(camera_frame_to_map,cam2map);
	    
	    cv_bridge::CvImageConstPtr bridge;
	    try
	    {
		bridge = cv_bridge::toCvCopy(msg, "32FC1");
	    }
	    catch (cv_bridge::Exception& e)
	    {
		ROS_ERROR("Failed to transform depth image.");
		return;
	    }

	    if(frame_counter_ < 3){++frame_counter_; return;}
	    
	    tracker_m.lock();
	    if(!myTracker_->Quit())
	    {
		myTracker_->SetCurrentTransformation(cam2map.matrix());
		myTracker_->UpdateDepth(bridge->image);
		myTracker_->FuseDepth();
	    }
	    else 
	    {
		ros::shutdown();
	    }
	    tracker_m.unlock();
	}
	
	bool plan_grasp_callback(grasp_planner::PlanGrasp::Request  &req,
		grasp_planner::PlanGrasp::Response &res ) {


	    tf::StampedTransform object_frame_to_map;
	    try {
		tl.waitForTransform(object_map_frame_name, req.header.frame_id, ros::Time(0), ros::Duration(1.0) );
		tl.lookupTransform(object_map_frame_name, req.header.frame_id, ros::Time(0), object_frame_to_map);
	    } catch (tf::TransformException ex) {
		ROS_ERROR("%s",ex.what());
		return false;
	    }
	    Eigen::Affine3d obj2obj_fr, obj_fr2map_fr, obj2map;
	    Eigen::Affine3f obj2map_f;
	    tf::transformTFToEigen(object_frame_to_map,obj_fr2map_fr);
	    tf::poseMsgToEigen(req.objectPose,obj2obj_fr);
	    //obj2obj_fr = *obj2obj_fr;
	    obj2map = obj2obj_fr*obj_fr2map_fr*Eigen::AngleAxisd(M_PI/2,Eigen::Vector3d::UnitX());
	    obj2map_f = obj2map.cast<float>(); //.setIdentity(); //
	    tf::transformEigenToTF(obj2map, gripper2map);
	    grasp_frame_set=true;

	    CylinderConstraint cc(obj2map_f,req.object_radius,req.object_height);
	    tracker_m.lock();
	    gripper_map->computeValidConfigs(myTracker_, cc);

	    //drawing
	    constraint_map::SimpleOccMapMsg msg2;
	    gripper_map->resetMap();
	    gripper_map->drawValidConfigsSmall();
	    //obj2map_f.setIdentity();
	    //gripper_map->drawCylinder(obj2map_f,req.object_radius,req.object_height);
	    gripper_map->updateMap();
	    gripper_map->toMessage(msg2);
	    msg2.header.frame_id = gripper_frame_name;
	    gripper_map_publisher_.publish(msg2);
	    tracker_m.unlock();


	}

	bool publish_map_callback(std_srvs::Empty::Request  &req,
		std_srvs::Empty::Response &res ) {
	    
	    ROS_INFO("Publishing maps");
#if 0
	    constraint_map::SimpleOccMapMsg msg;
	    tracker_m.lock();
	    myTracker_->toMessage(msg);
	    tracker_m.unlock();
	    msg.header.frame_id = object_map_frame_name;
	    object_map_publisher_.publish(msg); 
#endif
	    constraint_map::SimpleOccMapMsg msg2;
	    gripper_map->resetMap();
	    gripper_map->drawValidConfigsSmall();
	    gripper_map->updateMap();
	    gripper_map->toMessage(msg2);
	    msg2.header.frame_id = gripper_frame_name;
	    gripper_map_publisher_.publish(msg2);

	    return true;
	}

};

#endif
