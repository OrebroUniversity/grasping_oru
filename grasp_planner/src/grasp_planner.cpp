#include <grasp_planner/grasp_planner.h>


namespace grasp_planner {

GraspPlanner::GraspPlanner(SDF_Parameters& parameters) {

  publish_pc = false;
  nh_ = ros::NodeHandle("~");
  n_ = ros::NodeHandle();

  nh_.param<std::string>("gripper_file",gripper_fname,"full.cons");
  nh_.param<std::string>("grasp_frame_name",gripper_frame_name,"planned_grasp");
  nh_.param<std::string>("map_frame_name",object_map_frame_name,"map_frame");
  nh_.param<std::string>("map_topic",object_map_topic,"object_map");
  nh_.param<std::string>("gripper_map_topic",gripper_map_topic,"gripper_map");
  nh_.param<std::string>("sdf_map_topic",sdf_map_topic,"sdf_map");
  nh_.param<std::string>("fused_pc_topic",fused_pc_topic,"fused_pc");
  nh_.param<std::string>("dumpfile",dumpfile,"results.m");
  nh_.param<std::string>("LoadVolume", loadVolume_,"none");

  gripper_map = new ConstraintMap();
  bool success = gripper_map->loadGripperConstraints(gripper_fname.c_str());
  isInfoSet1 = false;
  isInfoSet2 = false;
  bool offlineTracker = false;

  if(!success) {
    ROS_ERROR("could not load gripper constraints file from %s",gripper_fname.c_str());
    ros::shutdown();
  }
	   
  //Parameters for SDF tracking 
  myParameters_ = parameters;
  //node specific parameters
  nh_.param("use_tf", use_tf_, false);
  nh_.param("runTrackerFromVolume", offlineTracker, false);
  nh_.param<std::string>("depth_topic_name", depth_topic_name_,"/camera/depth_registered/image");
  nh_.param<std::string>("depth_info_topic_name", depth_info_topic_name_,"/camera/depth_registered/camera_info");
  nh_.param<std::string>("depth_topic_name2", depth_topic_name2_,"none");
  nh_.param<std::string>("depth_info_topic_name2", depth_info_topic_name2_,"none");
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

  //	    nh_.param<double>("orientation_tolerance", orientation_tolerance, 0.5); //RADIAN
  nh_.param<int>("min_envelope_volume", MIN_ENVELOPE_VOLUME,5); //Number of configurations
  cylinder_tolerance = 0; //-0.015;
  plane_tolerance = 0; //0.005;

  myParameters_.resolution = gripper_map->getResolution();
  myTracker_ = new SDFTracker(myParameters_);

  if(loadVolume_.compare(std::string("none"))!=0 && offlineTracker)
    {
      ROS_INFO("Loading pre-recorded sdf volume from %s",loadVolume_.c_str());
      //Assume we are operating offline
      myTracker_->LoadSDF(loadVolume_);
    } 

  sdf_map_publisher_ = nh_.advertise<sdf_tracker_msgs::SDFMap> (sdf_map_topic,10);
  fused_pc_publisher_ = nh_.advertise<sensor_msgs::PointCloud2> (fused_pc_topic,10);

  depth_subscriber_ = n_.subscribe(depth_topic_name_, 1, &GraspPlanner::depthCallback, this);
  depth_camera_info_subscriber_ = n_.subscribe(depth_info_topic_name_, 1, &GraspPlanner::depthInfoCallback, this);
	   
  if(depth_topic_name2_!="none") { 
		depth_subscriber2_ = n_.subscribe(depth_topic_name2_, 1, &GraspPlanner::depthCallback2, this);
		depth_camera_info_subscriber2_ = n_.subscribe(depth_info_topic_name2_, 1, &GraspPlanner::depthInfoCallback2, this);
  }

  plan_grasp_serrver_ = nh_.advertiseService("plan_grasp", &GraspPlanner::planGraspCallback, this);
  publish_map_server_ = nh_.advertiseService("publish_map", &GraspPlanner::publishMapCallback, this);
  save_map_server_ = nh_.advertiseService("save_map", &GraspPlanner::saveMapCallback, this);
  map_to_edt_ = nh_.advertiseService("map_to_edt", &GraspPlanner::mapToEdtCallback, this);
  clear_map_server_ = nh_.advertiseService("clear_map", &GraspPlanner::clearMapCallback, this);
  load_volume_server_ = nh_.advertiseService("load_volume", &GraspPlanner::loadVolumeCallback, this);
  load_constraints_server_ = nh_.advertiseService("load_constraints", &GraspPlanner::loadConstraintsCallback, this);
  vis_pub_ = nh_.advertise<visualization_msgs::MarkerArray>( "sdf_marker", 10, true );
  constraint_pub_ = nh_.advertise<visualization_msgs::MarkerArray>( "constraint_marker", 10, true );
	    
  heartbeat_tf_ = nh_.createTimer(ros::Duration(0.05), &GraspPlanner::publishTF, this);

  frame_counter_ = 0;
  grasp_frame_set=false;
}

GraspPlanner::~GraspPlanner() {
  if( gripper_map != NULL ) {
		delete gripper_map;
  }
  if(myTracker_ != NULL) 
    {
      delete myTracker_;
    }
}

void GraspPlanner::publishMap(const ros::TimerEvent& event) {
  sdf_tracker_msgs::SDFMap mapMsg;
  myTracker_->toMessage(mapMsg);
  mapMsg.header.frame_id = object_map_frame_name;
  sdf_map_publisher_.publish(mapMsg);
}

void GraspPlanner::publishTF(const ros::TimerEvent& event) {
  if(grasp_frame_set) {
		br.sendTransform(tf::StampedTransform(gripper2map, ros::Time::now(), object_map_frame_name, gripper_frame_name));
  }
  // TODO: Fix this #if statement.
#if 0
  //This code publishes a transform in expected coordinates. 
  tf::Transform ee2grasp;
  Eigen::Affine3d ee2grasp_eigen;
  ee2grasp_eigen = Eigen::AngleAxisd(M_PI/2,Eigen::Vector3d::UnitZ())*Eigen::AngleAxisd(-M_PI/2,Eigen::Vector3d::UnitY());
  tf::transformEigenToTF(ee2grasp_eigen, ee2grasp);
  br.sendTransform(tf::StampedTransform(ee2grasp, ros::Time::now(), ee_frame_name, internal_grasp_fname));
#endif
}

void GraspPlanner::publishPC() {

  if(myTracker_ == NULL) return;

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
  }

  else {
    
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
		for (int i = 0; i < myTracker_->triangles_.size(); ++i)  {
      
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

void GraspPlanner::depthInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg)
{
  if(isInfoSet1) return;
  //set image size, focal length and center from camera info
	
  cam1params_.image_width=msg->width;
  cam1params_.image_height=msg->height; 
  cam1params_.fx=msg->K[0];
  cam1params_.fy=msg->K[4];
  cam1params_.cx=msg->K[2];
  cam1params_.cy=msg->K[5];
  isInfoSet1 = true;
  
  ROS_INFO("Parameters set for camera 1");
}

void GraspPlanner::depthInfoCallback2(const sensor_msgs::CameraInfo::ConstPtr& msg)
{
  if(isInfoSet2) return;
  //set image size, focal length and center from camera info
  cam2params_.image_width=msg->width;
  cam2params_.image_height=msg->height; 
  cam2params_.fx=msg->K[0];
  cam2params_.fy=msg->K[4];
  cam2params_.cx=msg->K[2];
  cam2params_.cy=msg->K[5];
  isInfoSet2 = true;
  
  ROS_INFO("Parameters set for camera 2");
}

void GraspPlanner::depthCallback(const sensor_msgs::Image::ConstPtr& msg)
{
  if(myTracker_ == NULL) return;
  camera_link_ = msg->header.frame_id;
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
  cv::Mat converted;
  try {
    
    bridge = cv_bridge::toCvCopy(msg, "32FC1");
    double scale_factor = 1;
		if(strncmp(msg->encoding.c_str(),"16UC1", msg->encoding.size())==0) {
      //kinect v2, convert to metric scale
      scale_factor = 0.001;
		}
		bridge->image.convertTo(converted,CV_32FC1,scale_factor);
    
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
	
  tracker_m.lock();
  if(!myTracker_->Quit())
  {
    ROS_INFO("GPLAN: Fusing frame");
    myTracker_->FuseDepth(converted, cam1params_, cam2map.matrix());
  }
  else 
  {
		ros::shutdown();
  }
  tracker_m.unlock();
}

void GraspPlanner::depthCallback2(const sensor_msgs::Image::ConstPtr& msg)
{
  if(myTracker_ == NULL) return;
  camera_link_ = msg->header.frame_id;
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
  cv::Mat converted;
  try
  {
		bridge = cv_bridge::toCvCopy(msg, "32FC1");
		double scale_factor = 1;
		if(strncmp(msg->encoding.c_str(),"16UC1", msg->encoding.size())==0) {
      //kinect v2, convert to metric scale
      scale_factor = 0.001;
		}
		bridge->image.convertTo(converted,CV_32FC1,scale_factor);

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
	    
  tracker_m.lock();
  if(!myTracker_->Quit())
  {
    ROS_INFO("GPLAN: Fusing frame from camera 2");
    myTracker_->FuseDepth(converted, cam2params_, cam2map.matrix());
  }
  else 
  {
		ros::shutdown();
  }
  tracker_m.unlock();
}

bool GraspPlanner::loadConstraintsCallback(grasp_planner::LoadResource::Request  &req,
                                           grasp_planner::LoadResource::Response &res ) {
  
  delete gripper_map;
  gripper_map = new ConstraintMap();
  bool success = gripper_map->loadGripperConstraints(req.name.c_str());
  
}

bool GraspPlanner::loadVolumeCallback(grasp_planner::LoadResource::Request  &req,
                                      grasp_planner::LoadResource::Response &res ) {
  
  if(myTracker_ == NULL) return false;
  tracker_m.lock();
  myTracker_->LoadSDF(req.name);
  tracker_m.unlock();

  return true;
}

bool GraspPlanner::mapToEdtCallback(std_srvs::Empty::Request  &req,
                                    std_srvs::Empty::Response &res ) {
  
  if(myTracker_ == NULL) return false;
  sdf_tracker_msgs::SDFMap mapMsg;
	
  tracker_m.lock();
  myTracker_->convertToEuclidean();
  myTracker_->toMessage(mapMsg);
  tracker_m.unlock();
	
  mapMsg.header.frame_id = object_map_frame_name;
  sdf_map_publisher_.publish(mapMsg);
  return true;
}

bool GraspPlanner::saveMapCallback(std_srvs::Empty::Request  &req,
                                   std_srvs::Empty::Response &res ) {
  
  if(myTracker_ == NULL) return false;
  tracker_m.lock();
  myTracker_->SaveSDF();
  tracker_m.unlock();
  return true;
}

bool GraspPlanner::clearMapCallback(std_srvs::Empty::Request  &req,
                                    std_srvs::Empty::Response &res ) {
  
  if(myTracker_ == NULL) return false;
  tracker_m.lock();
  myTracker_->ResetSDF();
  tracker_m.unlock();
  return true;
}

bool GraspPlanner::publishMapCallback(std_srvs::Empty::Request  &req,
                                      std_srvs::Empty::Response &res ) {	   
  ros::TimerEvent ev; 
  this->publishPC();
  //this->publishMap(ev);
  return true;
}

bool GraspPlanner::planGraspCallback(grasp_planner::PlanGrasp::Request  &req,
                                     grasp_planner::PlanGrasp::Response &res ) {

  if(myTracker_ == NULL) return false;
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
	    
  tf::StampedTransform prototype_in_plan_frame;
  try {
		tl.waitForTransform(gripper_frame_name, req.approach_frame, ros::Time(0), ros::Duration(1.0) );
		tl.lookupTransform( gripper_frame_name, req.approach_frame, ros::Time(0), prototype_in_plan_frame);
  } catch (tf::TransformException ex) {
		ROS_ERROR("%s",ex.what());
		return false;
  }
  Eigen::Affine3d proto2grasp_d;
  Eigen::Affine3f proto2grasp;
  tf::transformTFToEigen(prototype_in_plan_frame, proto2grasp_d);
  proto2grasp = proto2grasp_d.cast<float>();

  Eigen::Vector3f prototype;
  prototype<<req.approach_vector[0],req.approach_vector[1],req.approach_vector[2];
  prototype = proto2grasp.rotation()*prototype;

  std::cout<<prototype.transpose()<<std::endl;
  //std::cout<<"EE in grasp frame is at:\n"<<ee2grasp.matrix()<<std::endl;
  //CylinderConstraint cc(obj2map_f,req.object_radius,req.object_height);
  GripperPoseConstraint out;
  tracker_m.lock();
  ROS_INFO("[PLAN GRASP] Got lock");
  gripper_map->computeValidConfigs(myTracker_, obj2map_f, req.object_radius, req.object_height, prototype, req.approach_angle, out);
  tracker_m.unlock();
	  
  res.min_oa = out.min_oa;
  res.max_oa = out.max_oa; 
  res.volume=out.cspace_volume;
  res.time= out.debug_time;

  ROS_INFO("Publishing results");

  Eigen::Vector3f normal;
  float d = 0;

  //fill in results
  //order is: bottom, top, left, right, inner, outer
  hiqp_msgs::Primitive bottom, top, left, right, inner, outer;
  // TODO: Make sure grasping is done in this frame.
  const std::string grasping_frame = "world";

  // bottom
  normal = grasp2global.rotation()*out.lower_plane.a;
  d = out.lower_plane.b+normal.dot(grasp2global.translation());

  bottom.name = "bottom"; bottom.type = "plane"; bottom.frame_id = grasping_frame; // TODO: Check this.
  bottom.visible = true;
  bottom.color = {0.0, 0.0, 1.0, 0.2};
  bottom.parameters = { normal(0), normal(1), normal(2), d + plane_tolerance };
  
  //top 
  normal = grasp2global.rotation()*out.upper_plane.a;
  d = out.upper_plane.b+normal.dot(grasp2global.translation());

  top.name = "top"; top.type = "plane"; top.frame_id = grasping_frame; // TODO: Check this.
  top.visible = true;
  top.color = {0.0, 0.0, 1.0, 0.2};
  top.parameters = { -normal(0), -normal(1), -normal(2), - d - plane_tolerance };

  //left
  normal = grasp2global.rotation()*out.left_bound_plane.a;
  d = out.left_bound_plane.b+normal.dot(grasp2global.translation());

  left.name = "left"; left.type = "plane"; left.frame_id = grasping_frame; // TODO: Check this.
  left.visible = true;
  left.color = {0.0, 0.0, 1.0, 0.2};
  left.parameters = { -normal(0), -normal(1), -normal(2), -d - plane_tolerance };

  //right
  normal = grasp2global.rotation()*out.right_bound_plane.a;
  d = out.right_bound_plane.b+normal.dot(grasp2global.translation());

  right.name = "right"; right.type = "plane"; right.frame_id = grasping_frame; // TODO: Check this.
  right.visible = true;
  right.color = {0.0, 0.0, 1.0, 0.2};
  right.parameters = { normal(0), normal(1), normal(2), d + plane_tolerance };

  Eigen::Vector3f zaxis = grasp2global.rotation()*out.inner_cylinder.pose*Eigen::Vector3f::UnitZ();
  
  if(!out.isSphere) {
    
		//inner
    inner.name = "inner"; inner.type = "cylinder"; inner.frame_id = grasping_frame; // TODO: Check this.
    inner.visible = true;
    inner.color = {0.0, 0.0, 1.0, 0.2};
    inner.parameters = {zaxis(0), zaxis(1), zaxis(2), // Axis first.
                        grasp2global.translation()(0)+out.inner_cylinder.pose.translation()(0), // Position x
                        grasp2global.translation()(1)+out.inner_cylinder.pose.translation()(1), // Position y
                        grasp2global.translation()(2)+out.inner_cylinder.pose.translation()(2), // Position z
                        out.inner_cylinder.radius_ - cylinder_tolerance, // radius
                        req.object_height}; // height - shouldn't really matter unless the gripper is "YUGE".
                          

    //outer
    zaxis = out.outer_cylinder.pose*Eigen::Vector3f::UnitZ();
    outer.name = "outer"; outer.type = "cylinder"; outer.frame_id = grasping_frame;
    outer.visible = true;
    outer.color = {0.0, 0.0, 1.0, 0.2};
    outer.parameters = {zaxis(0), zaxis(1), zaxis(2),
                        grasp2global.translation()(0)+out.outer_cylinder.pose.translation()(0),
                        grasp2global.translation()(1)+out.outer_cylinder.pose.translation()(1),
                        grasp2global.translation()(2)+out.outer_cylinder.pose.translation()(2),
                        out.outer_cylinder.radius_ - cylinder_tolerance,
                        req.object_height}; // TODO: verify.

  } else {
    // outer
    outer.name = "outer"; outer.type = "sphere"; outer.frame_id = grasping_frame;
    outer.visible = true; outer.color = {0.0, 0.0, 1.0, 0.2};
    outer.parameters = {grasp2global.translation()(0)+out.outer_sphere.center(0),
                        grasp2global.translation()(1)+out.outer_sphere.center(1),
                        grasp2global.translation()(2)+out.outer_sphere.center(2),
                        out.outer_sphere.radius - cylinder_tolerance};

    inner.name = "inner"; inner.type = "sphere"; inner.frame_id = grasping_frame;
    inner.visible = true; inner.color = {0.0, 0.0, 1.0, 0.2};
    inner.parameters = {grasp2global.translation()(0)+out.inner_sphere.center(0),
                        grasp2global.translation()(1)+out.inner_sphere.center(1),
                        grasp2global.translation()(2)+out.inner_sphere.center(2),
                        out.inner_sphere.radius - cylinder_tolerance};
		
  }
  res.frame_id = req.header.frame_id;
  // NOTE: C++11 aggregate initialization.
  res.constraints = {bottom, top, left, right, inner, outer};
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
	    
	    
  addPlaneMarker(marker_array, -out.left_bound_plane.a.cast<double>(), -out.left_bound_plane.b-plane_tolerance, gripper_frame_name);
	    
  addPlaneMarker(marker_array, out.right_bound_plane.a.cast<double>(), out.right_bound_plane.b+plane_tolerance, gripper_frame_name);
  if(!out.isSphere) {
		zaxis = out.inner_cylinder.pose*Eigen::Vector3f::UnitZ();
		Eigen::Vector3d tmpose;
		tmpose = (out.inner_cylinder.pose.translation()).cast<double>();
		tmpose(2) += out.lower_plane.b-plane_tolerance;
		addCylinderMarker(marker_array, tmpose, zaxis.cast<double>(), out.inner_cylinder.radius_-cylinder_tolerance, gripper_frame_name, 
                      out.upper_plane.b-out.lower_plane.b+2*plane_tolerance);

		zaxis = out.outer_cylinder.pose*Eigen::Vector3f::UnitZ();
		addCylinderMarker(marker_array, tmpose, zaxis.cast<double>(), out.outer_cylinder.radius_-cylinder_tolerance, gripper_frame_name,
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

void GraspPlanner::addPlaneMarker(visualization_msgs::MarkerArray& markers,
                                  Eigen::Vector3d n,
                                  double d,
                                  std::string frame_, 
                                  std::string namespc,
                                  double r,
                                  double g,
                                  double b)
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

void GraspPlanner::addCylinderMarker(visualization_msgs::MarkerArray& markers,
                                     Eigen::Vector3d p,
                                     Eigen::Vector3d v,
                                     double r,
                                     std::string frame_,
                                     double h,
                                     std::string namespc,
                                     double rc,
                                     double g,
                                     double b)
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
void GraspPlanner::addSphereMarker(visualization_msgs::MarkerArray& markers,
                                   Eigen::Vector3f center,
                                   float radius,
                                   std::string frame_,
                                   std::string namespc,
                                   double r,
                                   double g,
                                   double b)
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


}
