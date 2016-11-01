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
#include <grasp_planner/LoadResource.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#define POINT_SCALE  0.02
#define LINE_SCALE   0.3
#define PLANE_SCALE  1
#define CONE_SCALE   0.3
#define LINE_WIDTH   0.005


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
	Eigen::Affine3d cam2map, prev_cam2map;
	
	//ros::Publisher gripper_map_publisher_;
	ros::Publisher fused_pc_publisher_;
	ros::Publisher vis_pub_;
	ros::Publisher constraint_pub_;
	ros::Subscriber depth_subscriber_;
	ros::Timer heartbeat_tf_;
//	ros::Timer heartbeat_pc_;

	ros::ServiceServer plan_grasp_serrver_;
	ros::ServiceServer publish_map_server_;
	ros::ServiceServer save_map_server_;
	ros::ServiceServer clear_map_server_;
	ros::ServiceServer load_volume_server_;
	ros::ServiceServer load_constraints_server_;

	tf::TransformBroadcaster br;
	tf::TransformListener tl;
	tf::Transform gripper2map;

	std::string gripper_fname;
	std::string gripper_frame_name;
	std::string ee_frame_name;
	std::string object_map_frame_name;
	std::string gripper_map_topic;
	std::string object_map_topic;
	std::string depth_topic_name_;
	std::string camera_link_;
	std::string fused_pc_topic;
	std::string loadVolume_;
	std::string dumpfile;

	int skip_frames_, frame_counter_;
	bool use_tf_, grasp_frame_set, publish_pc;
	double cylinder_tolerance, plane_tolerance, orientation_tolerance;
	int MIN_ENVELOPE_VOLUME;

    public:
	GraspPlannerNode(SDF_Parameters &parameters) {

	    publish_pc = false;
	    nh_ = ros::NodeHandle("~");
	    n_ = ros::NodeHandle();

	    nh_.param<std::string>("gripper_file",gripper_fname,"full.cons");
	    nh_.param<std::string>("grasp_frame_name",gripper_frame_name,"planned_grasp");
	    nh_.param<std::string>("ee_frame_name",ee_frame_name,"velvet_fingers_palm");
	    nh_.param<std::string>("map_frame_name",object_map_frame_name,"map_frame");
	    nh_.param<std::string>("map_topic",object_map_topic,"object_map");
	    nh_.param<std::string>("gripper_map_topic",gripper_map_topic,"gripper_map");
	    nh_.param<std::string>("fused_pc_topic",fused_pc_topic,"fused_pc");
	    nh_.param<std::string>("dumpfile",dumpfile,"results.m");
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

	    nh_.param<double>("orientation_tolerance", orientation_tolerance, 0.5); //RADIAN
	    nh_.param<int>("min_envelope_volume", MIN_ENVELOPE_VOLUME,5); //Number of configurations
	    cylinder_tolerance = -0.015;
	    plane_tolerance = 0.005;

	    myParameters_.resolution = gripper_map->getResolution();
	    myTracker_ = new SDFTracker(myParameters_);

	    if(loadVolume_.compare(std::string("none"))!=0)
	    {
		myTracker_->LoadSDF(loadVolume_);
	    }
	    
	    //subscribe / advertise
	    //gripper_map_publisher_ = nh_.advertise<constraint_map::SimpleOccMapMsg> (gripper_map_topic,10);
	    fused_pc_publisher_ = nh_.advertise<sensor_msgs::PointCloud2> (fused_pc_topic,10);

	    depth_subscriber_ = n_.subscribe(depth_topic_name_, 1, &GraspPlannerNode::depthCallback, this);
	    plan_grasp_serrver_ = nh_.advertiseService("plan_grasp", &GraspPlannerNode::plan_grasp_callback, this);
	    publish_map_server_ = nh_.advertiseService("publish_map", &GraspPlannerNode::publish_map_callback, this);
	    save_map_server_ = nh_.advertiseService("save_map", &GraspPlannerNode::save_map_callback, this);
	    clear_map_server_ = nh_.advertiseService("clear_map", &GraspPlannerNode::clear_map_callback, this);
	    load_volume_server_ = nh_.advertiseService("load_volume", &GraspPlannerNode::load_volume_callback, this);
	    load_constraints_server_ = nh_.advertiseService("load_constraints", &GraspPlannerNode::load_constraints_callback, this);
	    vis_pub_ = nh_.advertise<visualization_msgs::MarkerArray>( "sdf_marker", 10, true );
	    constraint_pub_ = nh_.advertise<visualization_msgs::MarkerArray>( "constraint_marker", 10, true );
	    
	    heartbeat_tf_ = nh_.createTimer(ros::Duration(0.1), &GraspPlannerNode::publishTF, this);
	    //heartbeat_pc_ = nh_.createTimer(ros::Duration(60), &GraspPlannerNode::publishPC, this);

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
        void publishPC() {

	    if(publish_pc) {
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
	    } else {

		//publish triangles instead	    
		visualization_msgs::MarkerArray marker_array;
		visualization_msgs::Marker marker;
		marker.header.frame_id = object_map_frame_name;
		marker.header.stamp = ros::Time::now();
		marker.ns = "my_namespace";
		marker.id = 0;
		marker.type = visualization_msgs::Marker::TRIANGLE_LIST;
		marker.action = visualization_msgs::Marker::ADD;
		marker.pose.position.x = 0;
		marker.pose.position.y = 0;
		marker.pose.position.z = 0;
		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;
		marker.pose.orientation.z = 0.0;
		marker.pose.orientation.w = 1.0;
		marker.scale.x = 1;
		marker.scale.y = 1;
		marker.scale.z = 1;
		marker.color.a = 1.0;
		marker.color.r = 0.0;
		marker.color.g = 1.0;
		marker.color.b = 0.0;

		ROS_INFO("Generating Triangles");
		tracker_m.lock();
		ROS_INFO("Got lock");
		myTracker_->triangles_.clear();
		myTracker_->MakeTriangles();
		for (int i = 0; i < myTracker_->triangles_.size(); ++i)
		{

		    Eigen::Vector3d vcolor;
		    vcolor << fabs(myTracker_->SDFGradient(myTracker_->triangles_[i],2,0)),
			   fabs(myTracker_->SDFGradient(myTracker_->triangles_[i],2,1)),
			   fabs(myTracker_->SDFGradient(myTracker_->triangles_[i],2,2));

		    std_msgs::ColorRGBA color;

		    /* normal-mapped color */
		     vcolor.normalize();
		     color.r = float(0.8*fabs(vcolor(2))+0.2*fabs(vcolor(1)));
		     color.g = float(0.8*fabs(vcolor(2))+0.2*fabs(vcolor(1)));
		     color.b = float(0.8*fabs(vcolor(2))+0.2*fabs(vcolor(1)));

		    color.a = 1.0f;

		    geometry_msgs::Point vertex;
		    vertex.x = myTracker_->triangles_[i](0);
		    vertex.y = myTracker_->triangles_[i](1);
		    vertex.z = myTracker_->triangles_[i](2);
		    marker.points.push_back(vertex);
		    marker.colors.push_back(color);
		}
		tracker_m.unlock();    
		ROS_INFO("Publishing Markers");
		
		marker_array.markers.push_back(marker);
		vis_pub_.publish( marker_array );
	    }
	}

	void depthCallback(const sensor_msgs::Image::ConstPtr& msg)
	{
	    tf::StampedTransform camera_frame_to_map;
	    try {
		tl.waitForTransform(object_map_frame_name, camera_link_, msg->header.stamp, ros::Duration(0.15) );
		tl.lookupTransform(object_map_frame_name,camera_link_, msg->header.stamp, camera_frame_to_map);
	    } catch (tf::TransformException ex) {
		ROS_ERROR("%s",ex.what());
		return;
	    }
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

	    if(frame_counter_ < 3) {
		++frame_counter_; 
		prev_cam2map = cam2map;
		return;
	    }

	    Eigen::Affine3d Tmotion;
	    Tmotion = prev_cam2map.inverse()*cam2map;
	    Eigen::AngleAxisd ax(Tmotion.rotation());
	    /*if(Tmotion.translation().norm() <0.01 && ax.angle()< 0.01) {
		//ROS_INFO("skipping frame %lf, %lf",Tmotion.translation().norm(),ax.angle());
		return;
	    } else {
		prev_cam2map = cam2map;
	    }*/
	    
	    tracker_m.lock();
	    if(!myTracker_->Quit())
	    {
		if(frame_counter_ < 50) {
		    ++frame_counter_; 
		} else {
		    //FIXME: throttling down fusing here
		    frame_counter_ = 3;
		ROS_INFO("GPLAN: Fusing frame");
		myTracker_->SetCurrentTransformation(cam2map.matrix());
		myTracker_->UpdateDepth(bridge->image);
		myTracker_->FuseDepth();
		}
	//	ROS_INFO("DONE");
	    }
	    else 
	    {
		ros::shutdown();
	    }
	    tracker_m.unlock();
	}
	
	bool load_constraints_callback(grasp_planner::LoadResource::Request  &req,
		grasp_planner::LoadResource::Response &res ) {

	    delete gripper_map;
	    gripper_map = new ConstraintMap();
	    bool success = gripper_map->loadGripperConstraints(req.name.c_str());

	}

	bool load_volume_callback(grasp_planner::LoadResource::Request  &req,
		grasp_planner::LoadResource::Response &res ) {
	    
	    tracker_m.lock();
	    myTracker_->LoadSDF(req.name);
	    tracker_m.unlock();

	    return true;
	}
	
	bool save_map_callback(std_srvs::Empty::Request  &req,
		std_srvs::Empty::Response &res ) {
	    
	    tracker_m.lock();
	    myTracker_->SaveSDF();
	    tracker_m.unlock();
	    return true;
	}
	
	bool clear_map_callback(std_srvs::Empty::Request  &req,
		std_srvs::Empty::Response &res ) {
	    
	    tracker_m.lock();
	    myTracker_->ResetSDF();
	    tracker_m.unlock();
	    return true;
	}
	
	bool publish_map_callback(std_srvs::Empty::Request  &req,
		std_srvs::Empty::Response &res ) {
	    
	    this->publishPC();
	    return true;
	}

	bool plan_grasp_callback(grasp_planner::PlanGrasp::Request  &req,
		grasp_planner::PlanGrasp::Response &res ) {

	    //FIXME: this slows us down, but it helps with debugging/visualization
	    //this->publishPC();
	    ROS_INFO("Got request");
	    
	    std::cout<<"From frame "<<req.header.frame_id<<" to "<<object_map_frame_name<<std::endl;
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
	    Eigen::Affine3f grasp2global;

	    tf::transformTFToEigen(object_frame_to_map,obj_fr2map_fr);
	    tf::poseMsgToEigen(req.objectPose,obj2obj_fr);
	    //obj2obj_fr = *obj2obj_fr;
	    obj2map = obj_fr2map_fr*obj2obj_fr; //*Eigen::AngleAxisd(M_PI/2,Eigen::Vector3d::UnitX())*Eigen::AngleAxisd(M_PI,Eigen::Vector3d::UnitZ());
	    obj2map_f = obj2map.cast<float>(); //.setIdentity(); //
	    tf::transformEigenToTF(obj2map, gripper2map);
	    grasp2global = obj2obj_fr.cast<float>();

	    grasp_frame_set=true;
	    
	    tf::StampedTransform ee_in_plan_frame;
	    try {
		tl.waitForTransform(gripper_frame_name, ee_frame_name, ros::Time(0), ros::Duration(1.0) );
		tl.lookupTransform( gripper_frame_name, ee_frame_name, ros::Time(0), ee_in_plan_frame);
	    } catch (tf::TransformException ex) {
		ROS_ERROR("%s",ex.what());
		return false;
	    }
	    Eigen::Affine3d ee2grasp_d;
	    Eigen::Affine3f ee2grasp;
	    tf::transformTFToEigen(ee_in_plan_frame, ee2grasp_d);
	    ee2grasp = ee2grasp_d.cast<float>();

	    //std::cout<<"EE in grasp frame is at:\n"<<ee2grasp.matrix()<<std::endl;
	    //CylinderConstraint cc(obj2map_f,req.object_radius,req.object_height);
	    GripperPoseConstraint out;
	    tracker_m.lock();
	    ROS_INFO("[PLAN GRASP] Got lock");
	    gripper_map->computeValidConfigs(myTracker_, obj2map_f, req.object_radius, req.object_height, ee2grasp, orientation_tolerance, out);
	    tracker_m.unlock();
	  
	    res.min_oa = out.min_oa;
	    res.max_oa = out.max_oa; 
	    res.volume=out.cspace_volume;
	    res.time= out.debug_time;

	    ROS_INFO("Publishing results");

	    //fill in results
	    //order is: bottom, top, left, right, inner, outer
	    hqp_controllers_msgs::TaskGeometry task;
	    task.g_type = hqp_controllers_msgs::TaskGeometry::PLANE;
	   
	    Eigen::Vector3f normal;
	    float d = 0;

	    //bottom 
	    normal = grasp2global.rotation()*out.lower_plane.a;
	    d = out.lower_plane.b+normal.dot(grasp2global.translation());
	    task.g_data.clear();	    
	    task.g_data.push_back(-normal(0));
	    task.g_data.push_back(-normal(1));
	    task.g_data.push_back(-normal(2));
	    task.g_data.push_back(-d-plane_tolerance);
	    res.constraints.push_back(task);
	    //top 
	    normal = grasp2global.rotation()*out.upper_plane.a;
	    d = out.upper_plane.b+normal.dot(grasp2global.translation());
	    task.g_data.clear();	    
	    task.g_data.push_back(normal(0));
	    task.g_data.push_back(normal(1));
	    task.g_data.push_back(normal(2));
	    task.g_data.push_back(d+plane_tolerance);
	    res.constraints.push_back(task);
	    //left
	    normal = grasp2global.rotation()*out.left_bound_plane.a;
	    d = out.left_bound_plane.b+normal.dot(grasp2global.translation());
	    task.g_data.clear();	    
	    task.g_data.push_back(-normal(0));
	    task.g_data.push_back(-normal(1));
	    task.g_data.push_back(-normal(2));
	    task.g_data.push_back(-d-plane_tolerance);
	    res.constraints.push_back(task);
	    //right
	    normal = grasp2global.rotation()*out.right_bound_plane.a;
	    d = out.right_bound_plane.b+normal.dot(grasp2global.translation());
	    task.g_data.clear();	    
	    task.g_data.push_back(normal(0));
	    task.g_data.push_back(normal(1));
	    task.g_data.push_back(normal(2));
	    task.g_data.push_back(d+plane_tolerance);
	    res.constraints.push_back(task);
	    

	    Eigen::Vector3f zaxis = grasp2global.rotation()*out.inner_cylinder.pose*Eigen::Vector3f::UnitZ();
	    if(!out.isSphere) {
		task.g_type = hqp_controllers_msgs::TaskGeometry::CYLINDER;
		//inner
		task.g_data.clear();	    
		task.g_data.push_back(grasp2global.translation()(0)+out.inner_cylinder.pose.translation()(0));
		task.g_data.push_back(grasp2global.translation()(1)+out.inner_cylinder.pose.translation()(1));
		task.g_data.push_back(grasp2global.translation()(2)+out.inner_cylinder.pose.translation()(2));
		task.g_data.push_back(zaxis(0));
		task.g_data.push_back(zaxis(1));
		task.g_data.push_back(zaxis(2));
		task.g_data.push_back(out.inner_cylinder.radius_ - cylinder_tolerance);
		res.constraints.push_back(task);
		//outer
		zaxis = out.outer_cylinder.pose*Eigen::Vector3f::UnitZ();
		task.g_data.clear();	    
		task.g_data.push_back(grasp2global.translation()(0)+out.outer_cylinder.pose.translation()(0));
		task.g_data.push_back(grasp2global.translation()(1)+out.outer_cylinder.pose.translation()(1));
		task.g_data.push_back(grasp2global.translation()(2)+out.outer_cylinder.pose.translation()(2));
		task.g_data.push_back(zaxis(0));
		task.g_data.push_back(zaxis(1));
		task.g_data.push_back(zaxis(2));
		task.g_data.push_back(out.outer_cylinder.radius_ - cylinder_tolerance);
		res.constraints.push_back(task);
	    } else {
		task.g_type = hqp_controllers_msgs::TaskGeometry::SPHERE;
		task.g_data.clear();	    
		task.g_data.push_back(grasp2global.translation()(0)+out.outer_sphere.center(0));
		task.g_data.push_back(grasp2global.translation()(1)+out.outer_sphere.center(1));
		task.g_data.push_back(grasp2global.translation()(2)+out.outer_sphere.center(2));
		task.g_data.push_back(out.outer_sphere.radius - cylinder_tolerance);
		res.constraints.push_back(task);

		task.g_data.clear();	    
		task.g_data.push_back(grasp2global.translation()(0)+out.inner_sphere.center(0));
		task.g_data.push_back(grasp2global.translation()(1)+out.inner_sphere.center(1));
		task.g_data.push_back(grasp2global.translation()(2)+out.inner_sphere.center(2));
		task.g_data.push_back(out.inner_sphere.radius - cylinder_tolerance);
		res.constraints.push_back(task);
		
	    }
	    res.frame_id = req.header.frame_id;
	    res.success = res.volume > MIN_ENVELOPE_VOLUME;

	    //Display functions
	    pcl::PointCloud<pcl::PointXYZRGB> pc;
	    gripper_map->getConfigsForDisplay(pc);
	    gripper_map->generateOpeningAngleDump(dumpfile);
	    
	    sensor_msgs::PointCloud2 cloud;
	    pcl::toROSMsg(pc,cloud);
	    cloud.header.frame_id = gripper_frame_name;
	    cloud.header.stamp = ros::Time::now();
	    fused_pc_publisher_.publish(cloud);
    
	    visualization_msgs::MarkerArray marker_array;
	    normal = grasp2global.rotation()*out.lower_plane.a;
	    d = out.lower_plane.b+normal.dot(grasp2global.translation());
	    addPlaneMarker(marker_array, normal.cast<double>(), d+plane_tolerance, req.header.frame_id);
	    normal = grasp2global.rotation()*out.upper_plane.a;
	    d = out.upper_plane.b+normal.dot(grasp2global.translation());
	    addPlaneMarker(marker_array, -normal.cast<double>(), -d-plane_tolerance, req.header.frame_id);
	    
	    //addPlaneMarker(marker_array, -out.upper_plane.a.cast<double>(), -out.upper_plane.b-plane_tolerance, gripper_frame_name);
	    
	    addPlaneMarker(marker_array, -out.left_bound_plane.a.cast<double>(), -out.left_bound_plane.b-plane_tolerance, gripper_frame_name);
	    
	    addPlaneMarker(marker_array, out.right_bound_plane.a.cast<double>(), out.right_bound_plane.b+plane_tolerance, gripper_frame_name);
	    if(!out.isSphere) {
		zaxis = out.inner_cylinder.pose*Eigen::Vector3f::UnitZ();
		Eigen::Vector3d tmpose;
		tmpose = (out.inner_cylinder.pose.translation()).cast<double>();
		tmpose(2) += out.lower_plane.b-plane_tolerance;
		addCylinderMarker(marker_array, tmpose, zaxis.cast<double>(), out.inner_cylinder.radius_+cylinder_tolerance, gripper_frame_name, 
			out.upper_plane.b-out.lower_plane.b+2*plane_tolerance);

		zaxis = out.outer_cylinder.pose*Eigen::Vector3f::UnitZ();
		addCylinderMarker(marker_array, tmpose, zaxis.cast<double>(), out.outer_cylinder.radius_+cylinder_tolerance, gripper_frame_name,
		       	out.upper_plane.b-out.lower_plane.b+2*plane_tolerance);
		//add request cylinder
		addCylinderMarker(marker_array, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(), req.object_radius, gripper_frame_name, req.object_height, "request", 0.8, 0.1, 0.1);

	    } else {
		addSphereMarker(marker_array, out.inner_sphere.center, out.inner_sphere.radius-cylinder_tolerance, gripper_frame_name);
		addSphereMarker(marker_array, out.outer_sphere.center, out.outer_sphere.radius-cylinder_tolerance, gripper_frame_name);
		addSphereMarker(marker_array, obj2map_f.translation(), req.object_radius, object_map_frame_name, "request", 0.8, 0.1, 0.1);
	    }
	    ROS_INFO("Publishing %lu markers",marker_array.markers.size());
	    constraint_pub_.publish(marker_array);

	    return true;
	}

	void addPlaneMarker(visualization_msgs::MarkerArray& markers, Eigen::Vector3d n, double d, std::string frame_, 
		std::string namespc="plane", double r=1, double g=0.2, double b=0.5)
	{

	    //transformation which points x in the plane normal direction
	    Eigen::Quaterniond q;
	    q.setFromTwoVectors(Eigen::Vector3d::UnitX() , n);

	    visualization_msgs::Marker marker;

	    //normal
	    marker.header.frame_id = frame_;
	    marker.header.stamp = ros::Time::now();
	    marker.ns = "normal";
	    marker.type =  visualization_msgs::Marker::ARROW;
	    marker.action = visualization_msgs::Marker::ADD;
	    //marker.lifetime = ros::Duration(0.1);
	    marker.id = markers.markers.size();
	    marker.pose.position.x = n(0) * d;
	    marker.pose.position.y = n(1) * d;
	    marker.pose.position.z = n(2) * d;
	    marker.pose.orientation.x = q.x();
	    marker.pose.orientation.y = q.y();
	    marker.pose.orientation.z = q.z();
	    marker.pose.orientation.w = q.w();
	    marker.scale.x = LINE_SCALE;
	    marker.scale.y = 0.05 * LINE_SCALE;
	    marker.scale.z = 0.05 * LINE_SCALE;
	    marker.color.r = 1.0;
	    marker.color.g = 0.0;
	    marker.color.b = 1.0;
	    marker.color.a = 1.0;
	    markers.markers.push_back(marker);

	    //plane
	    q.setFromTwoVectors(Eigen::Vector3d::UnitZ() ,n);
	    marker.ns = namespc;
	    marker.type = visualization_msgs::Marker::CUBE;
	    marker.id = markers.markers.size();
	    marker.pose.orientation.x = q.x();
	    marker.pose.orientation.y = q.y();
	    marker.pose.orientation.z = q.z();
	    marker.pose.orientation.w = q.w();
	    marker.scale.x = PLANE_SCALE;
	    marker.scale.y = PLANE_SCALE;
	    marker.scale.z = 0.0001;
	    marker.color.r = r;
	    marker.color.g = g;
	    marker.color.b = b;
	    marker.color.a = 0.4;
	    markers.markers.push_back(marker);
	}

	void addCylinderMarker(visualization_msgs::MarkerArray& markers, Eigen::Vector3d p, Eigen::Vector3d v, double r, std::string frame_, double h=LINE_SCALE, 
		std::string namespc="cylinder", double rc=0, double g=1, double b=1)
	{
	    Eigen::Quaterniond q;

	    //transformation which points z in the cylinder direction
	    q.setFromTwoVectors(Eigen::Vector3d::UnitZ(), v);

	    visualization_msgs::Marker marker;
	    marker.ns = namespc;
	    marker.header.frame_id = frame_;
	    marker.header.stamp = ros::Time::now();
	    marker.type =  visualization_msgs::Marker::CYLINDER;
	    marker.action = visualization_msgs::Marker::ADD;
	    //marker.lifetime = ros::Duration(0.1);
	    marker.id = markers.markers.size();
	    marker.pose.position.x = p(0) + v(0) * 0.5 * h;//LINE_SCALE;
	    marker.pose.position.y = p(1) + v(1) * 0.5 * h;//LINE_SCALE;
	    marker.pose.position.z = p(2) + v(2) * 0.5 * h;
	    marker.pose.orientation.x = q.x();
	    marker.pose.orientation.y = q.y();
	    marker.pose.orientation.z = q.z();
	    marker.pose.orientation.w = q.w();
	    marker.scale.x = 2 * r;
	    marker.scale.y = 2 * r;
	    marker.scale.z = h;
	    marker.color.r = rc;
	    marker.color.g = g;
	    marker.color.b = b;
	    marker.color.a = 0.5;
	    markers.markers.push_back(marker);
	}
	void addSphereMarker(visualization_msgs::MarkerArray& markers, Eigen::Vector3f center, float radius, std::string frame_, 
		std::string namespc="sphere", double r=0, double g=1, double b=1)
	{
	    visualization_msgs::Marker marker;

	    marker.ns = namespc;
	    marker.header.frame_id = frame_;
	    marker.header.stamp = ros::Time::now();
	    marker.type = visualization_msgs::Marker::SPHERE;
	    marker.action = visualization_msgs::Marker::ADD;
	    marker.id = markers.markers.size();
	    marker.pose.position.x = center(0);
	    marker.pose.position.y = center(1);
	    marker.pose.position.z = center(2);
	    marker.pose.orientation.x = 0.0;
	    marker.pose.orientation.y = 0.0;
	    marker.pose.orientation.z = 0.0;
	    marker.pose.orientation.w = 1.0;
	    marker.scale.x = 2 * radius;
	    marker.scale.y = 2 * radius;
	    marker.scale.z = 2 * radius;
	    marker.color.r = r;
	    marker.color.g = g;
	    marker.color.b = b;
	    marker.color.a = 0.5;

	    markers.markers.push_back(marker);
	}



};

#endif
