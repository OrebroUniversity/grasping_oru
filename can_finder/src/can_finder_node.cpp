#include <ros/ros.h>

#include <hqp_controllers_msgs/TaskGeometry.h>
#include <hqp_controllers_msgs/FindCanTask.h>
#include <sensor_msgs/PointCloud2.h>

#include<pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>

#include <tf/transform_listener.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>

#include <Eigen/Core>
//#include <boost/thread/mutex.hpp>

//#define VANILLA_CANS

class CanFinderNode {

    private:
	ros::NodeHandle nh_;
	ros::NodeHandle n_;

	ros::Subscriber pcloud_sub;
	ros::Publisher pcloud_pub_;
	ros::ServiceServer find_can_srv_;
        tf::TransformListener tl;

	pcl::PointCloud<pcl::PointXYZ> my_cloud;
	std::string pcloud_topic;
	std::string pcloud_frame_name;
	std::string world_frame;
	std::string palm_frame;
	double expected_floor_height;
	double expected_pallet_height;
	double height_cutoff;
	double eps;
	double angle_thresh;
	double eval_thresh;
	double max_dist;
	double max_x;
	int min_number_pts;
	//boost::mutex::cloud_mutex;
	double grow_cylinder_m, inner2outer, grow_plane_m;

	double dist_factor;
	Eigen::Affine3d world2palm;
    public:

	CanFinderNode() {
	    nh_ = ros::NodeHandle("~");
	    n_ = ros::NodeHandle();
	    nh_.param<std::string>("pcloud_topic", pcloud_topic,"/camera/depth/points");
	    nh_.param<std::string>("world_frame", world_frame,"world");
	    nh_.param<std::string>("palm_frame", palm_frame,"velvet_fingers_palm");
	    nh_.param("floor_height", expected_floor_height  ,0.0);
	    nh_.param("pallet_height", expected_pallet_height ,0.1);
	    nh_.param("max_dist", max_dist ,2.0);
	    nh_.param("max_x", max_x ,1.3);
	    nh_.param("dist_factor", dist_factor ,0.02);
	    
	    nh_.param("min_pts_cluster",min_number_pts,250);
	    nh_.param("pallet_height_tolerance",eps,0.03); 
	    nh_.param("plane_angle_tolerance",angle_thresh,10*M_PI/180); 
	    nh_.param("objects_max_height",height_cutoff,0.3);
	    nh_.param("cylinder_evals_thresh",eval_thresh,0.8);
	    nh_.param("grow_cylinder_m",grow_cylinder_m,0.25);
	    nh_.param("cyl2cyl_m",inner2outer,0.2);
	    nh_.param("grow_plane_m",grow_plane_m,0.05);
	    
	    pcloud_frame_name = "";

	    find_can_srv_ = nh_.advertiseService("find_cans", &CanFinderNode::find_cans, this);
	    pcloud_sub = n_.subscribe(pcloud_topic, 1, &CanFinderNode::cloudCallback, this);
	    pcloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("result_cloud", 10); 

	}
	
	void transformPointCloudInPlace(Eigen::Transform<double,3,Eigen::Affine,Eigen::ColMajor> &Tr, pcl::PointCloud<pcl::PointXYZ> &pc)
	{
	    Eigen::Transform<float,3,Eigen::Affine,Eigen::ColMajor> T = Tr.cast<float>();
	    for(unsigned int pit=0; pit<pc.points.size(); ++pit)
	    {
		Eigen::Map<Eigen::Vector3f> pt((float*)&pc.points[pit],3);
		pt = T*pt;
	    }
	}


	void cloudCallback(const sensor_msgs::PointCloud2 &msg) {
	    
	    pcloud_frame_name = msg.header.frame_id;
	    
	    pcl::fromROSMsg(msg, my_cloud);
	    tf::StampedTransform to_world_tf;
	    try {
		tl.waitForTransform(world_frame, pcloud_frame_name, ros::Time(0), ros::Duration(1.0) );
		tl.lookupTransform(world_frame, pcloud_frame_name, ros::Time(0), to_world_tf);
	    } catch (tf::TransformException ex) {
		ROS_ERROR("%s",ex.what());
		return;
	    }
	    //ROS_INFO("got pointcloud in world frame, points: %ld",my_cloud.points.size());
	    Eigen::Affine3d cam2world;
	    tf::transformTFToEigen(to_world_tf,cam2world);
	    //std::cerr<<"cam2world: "<<cam2world.matrix()<<std::endl;
	    pcl::PointCloud<pcl::PointXYZ> temp_cloud;
	    for(int i=0; i<my_cloud.points.size(); ++i) {
		Eigen::Vector3d pt;
		pt<<my_cloud.points[i].x,my_cloud.points[i].y,my_cloud.points[i].z;
		if(pt.norm()<max_dist) {
		    temp_cloud.points.push_back(my_cloud.points[i]);
		}
	    }
	    temp_cloud.is_dense=false;
	    temp_cloud.width =1;
	    temp_cloud.height = temp_cloud.points.size();
	    my_cloud = temp_cloud;
	    this->transformPointCloudInPlace(cam2world, my_cloud);
	    
	    tf::StampedTransform palm_world_tf;
	    try {
		tl.waitForTransform(world_frame, palm_frame, ros::Time(0), ros::Duration(1.0) );
		tl.lookupTransform(world_frame, palm_frame, ros::Time(0), palm_world_tf);
	    } catch (tf::TransformException ex) {
		ROS_ERROR("%s",ex.what());
		return;
	    }
	    //ROS_INFO("got pointcloud in world frame, points: %ld",my_cloud.points.size());
	    tf::transformTFToEigen(palm_world_tf,world2palm);
	
	}

	bool find_cans(hqp_controllers_msgs::FindCanTask::Request  &req,
		    hqp_controllers_msgs::FindCanTask::Response &res ) {
	    

	    hqp_controllers_msgs::TaskGeometry bottom_plane, top_plane, inner_cylinder, outer_cylinder;
	    pcl::PointCloud<pcl::PointXYZRGB> resultCloud;
	    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>());
	    *cloud = my_cloud;
	    ROS_INFO("cloud in service has %ld points",cloud->points.size());

	    Eigen::Vector3d normal;
	    Eigen::Vector2d bl, ur, pca1, pca2;
	    bl <<INT_MAX,INT_MAX;
	    ur <<-INT_MAX,-INT_MAX;
	    // Create the segmentation object
	    pcl::SACSegmentation<pcl::PointXYZ> seg;
	    // Optional
	    seg.setOptimizeCoefficients (true);
	    // Mandatory
	    seg.setModelType (pcl::SACMODEL_PLANE);
	    seg.setMethodType (pcl::SAC_RANSAC);
	    seg.setDistanceThreshold (0.01); //threshold on distance to plane

	    bool foundPalletPlane = false;

	    while (!foundPalletPlane) {
		pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZ>());
		seg.setInputCloud (cloud);
		seg.segment (*inliers, *coefficients);
		// Extract the planar inliers from the input cloud
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud (cloud);
		extract.setIndices (inliers);

		if (inliers->indices.size () == 0)
		{
		    PCL_ERROR ("Could not estimate a planar model for the given dataset.");
		    break;
		}

		normal<<coefficients->values[0],coefficients->values[1],coefficients->values[2];
		double norm = normal.norm();
		normal.normalize();

		double alpha = acos(normal.dot(Eigen::Vector3d::UnitZ()));
		if(fabsf(alpha) < angle_thresh) {
		    //we have a plane parallel to the floor
		    //check z intercept
		    double plane_z = (-coefficients->values[3]/coefficients->values[2])/norm;
		    ROS_INFO("Found a horizontal plane at height %lf",plane_z);
		    if(fabsf(plane_z - expected_pallet_height) < eps ) {
			ROS_INFO("Plane matches ours");
			///////
			bottom_plane.g_type = hqp_controllers_msgs::TaskGeometry::PLANE;
			bottom_plane.g_data.push_back(normal(0));
			bottom_plane.g_data.push_back(normal(1));
			bottom_plane.g_data.push_back(normal(2));
			bottom_plane.g_data.push_back(0.29);
			
			top_plane.g_type = hqp_controllers_msgs::TaskGeometry::PLANE;
			top_plane.g_data.push_back(-normal(0));
			top_plane.g_data.push_back(-normal(1));
			top_plane.g_data.push_back(-normal(2));
			///////
			//we found our plane
			Eigen::MatrixXd planeM (inliers->indices.size (),2);
			Eigen::Vector2d mean;
			pcl::PointXYZRGB pt;
			uint8_t r = 155, g = 155, b = 10; // Example: Red color
			uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
			float col = *reinterpret_cast<float*>(&rgb);

			for (size_t i = 0; i < inliers->indices.size (); ++i) {
			    pt.x = cloud->points[inliers->indices[i]].x;
			    pt.y = cloud->points[inliers->indices[i]].y;
			    pt.z = cloud->points[inliers->indices[i]].z;
			    pt.rgb = col;
			    Eigen::Vector2d tmp;
			    tmp<<pt.x,pt.y;
			    mean += tmp;
			    resultCloud.points.push_back(pt);
			}
			mean /= (inliers->indices.size());
			for (size_t i = 0; i < inliers->indices.size (); ++i) {
			    pt.x = cloud->points[inliers->indices[i]].x;
			    pt.y = cloud->points[inliers->indices[i]].y;
			    planeM(i,0) = pt.x-mean(0);
			    planeM(i,1) = pt.y-mean(1);
			}
			Eigen::Matrix<double,2,2> covSum_ = planeM.transpose()*planeM;
			Eigen::Matrix<double,2,2> cov_ = covSum_/(inliers->indices.size()-1);

			//do PCA
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov_);
			std::cout<<"Plane cov: "<<cov_<<std::endl;
			std::cout<<"evals: "<<eig.eigenvalues()<<std::endl;
		        std::cout<<"evecs: "<<eig.eigenvectors()<<std::endl;	
			if(eig.eigenvalues()(0) < 0.01 || eig.eigenvalues()(1) < 0.01) {
			    ROS_WARN("PCA failed on plane, reguralizing");
			    double reg_factor = 0.1;
			    cov_ = cov_ + reg_factor*Eigen::Matrix2d::Identity();
			    eig.compute(cov_);
			    if(eig.eigenvalues()(0) < 0.01 || eig.eigenvalues()(1) < 0.01) {
				ROS_ERROR("Failed again, reverting to axis oriented");
				cov_=Eigen::Matrix2d::Identity();
				eig.compute(cov_);
			    }
			} 
			pca1 = eig.eigenvectors().col(0);
			pca2 = eig.eigenvectors().col(1);
			for (size_t i = 0; i < inliers->indices.size (); ++i) {
			    Eigen::Vector2d proj;
			    Eigen::Vector2d point;
			    point<<cloud->points[inliers->indices[i]].x,cloud->points[inliers->indices[i]].y;
			    proj(0) = point.dot(pca1);
			    proj(1) = point.dot(pca2);
			    if(proj(0) < bl(0)) bl(0) = proj(0);
			    if(proj(1) < bl(1)) bl(1) = proj(1);
			    if(proj(0) > ur(0)) ur(0) = proj(0);
			    if(proj(1) > ur(1)) ur(1) = proj(1);
			}



			// Remove the planar inliers, extract the rest
			extract.setNegative (true);
			extract.filter (*cloud_f);
			*cloud = *cloud_f;
			pcl::PointCloud<pcl::PointXYZ> ptemp;
			r = 15, g = 15, b = 250; 
			rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
			col = *reinterpret_cast<float*>(&rgb);
			for(int i=0; i<cloud->points.size(); ++i) {
			    if(cloud->points[i].z < plane_z || cloud->points[i].z > plane_z + height_cutoff) continue;
			    Eigen::Vector2d onplane;
			    onplane(0) = cloud->points[i].x;
			    onplane(1) = cloud->points[i].y;
			    Eigen::Vector2d proj;
			    proj(0) = onplane.dot(pca1);
			    proj(1) = onplane.dot(pca2);
			    if(proj(0) > bl(0) && proj(1) > bl(1) && proj(0)<ur(0) && proj(1)<ur(1)) {
				ptemp.points.push_back(cloud->points[i]);
				pcl::PointXYZRGB pt;
				pt.x = cloud->points[i].x;
				pt.y = cloud->points[i].y;
				pt.z = cloud->points[i].z;
				pt.rgb = col;
				resultCloud.points.push_back(pt);
			    }
			}
			ROS_INFO("Points on pallet: %ld",ptemp.points.size());
			foundPalletPlane = true;
			*cloud = ptemp; 
			

			break;

		    }	
		}

		//we didn't find our plane, remove inliers from cloud and try again

		std::cerr << "Model coefficients: " << coefficients->values[0] << " " 
		    << coefficients->values[1] << " "
		    << coefficients->values[2] << " " 
		    << coefficients->values[3] << std::endl;

		std::cerr << "Model inliers: " << inliers->indices.size () << std::endl;
		pcl::PointXYZRGB pt;
		uint8_t r = 255, g = 5, b = 0; // Example: Red color
		uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
		float col = *reinterpret_cast<float*>(&rgb);

		for (size_t i = 0; i < inliers->indices.size (); ++i) {
		    pt.x = cloud->points[inliers->indices[i]].x;
		    pt.y = cloud->points[inliers->indices[i]].y;
		    pt.z = cloud->points[inliers->indices[i]].z;
		    pt.rgb = col;
		    resultCloud.points.push_back(pt);
		}
		// Remove the planar inliers, extract the rest
		extract.setNegative (true);
		extract.filter (*cloud_f);
		*cloud = *cloud_f;
	    }

	    if(foundPalletPlane) {
		// Creating the KdTree object for the search method of the extraction
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
		tree->setInputCloud (cloud);

#ifdef VANILLA_CANS
		std::vector<pcl::PointIndices> cluster_indices;
		pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
		ec.setClusterTolerance (0.02); // 2cm
		ec.setMinClusterSize (min_number_pts);
		ec.setMaxClusterSize (25000);
		ec.setSearchMethod (tree);
		ec.setInputCloud (cloud);
		ec.extract (cluster_indices);

		int j = 0;
		for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
		{
		    Eigen::MatrixXd planeM (it->indices.size (),3);
		    Eigen::Vector3d mean;
		    mean<<0,0,0;
		    uint8_t r = 0, g = 55 + ((double)j/cluster_indices.size())*200., b = 0; // Example: Red color
		    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
		    float col = *reinterpret_cast<float*>(&rgb);
		    pcl::PointXYZRGB pt;
		    pt.rgb = col;
		    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit) {
			pt.x = cloud->points[*pit].x;
			pt.y = cloud->points[*pit].y;
			pt.z = cloud->points[*pit].z;
			Eigen::Vector3d tmp;
			tmp<<pt.x,pt.y,pt.z;
			mean += tmp;
		    }
		    mean /= (it->indices.size ());
		   
		    if(mean(0) > max_x) {
		       ROS_WARN("Ignoring far away cluster\n");
		       continue;
		    }	       
		    double max_z = -INT_MAX;
		    for (size_t i = 0; i < it->indices.size (); ++i) {
			Eigen::Vector3d tmp;
			tmp<<cloud->points[it->indices[i]].x,cloud->points[it->indices[i]].y,cloud->points[it->indices[i]].z;
			if(tmp(2) > max_z) max_z = tmp(2);
			planeM(i,0) = tmp(0) - mean(0);
			planeM(i,1) = tmp(1) - mean(1);
			planeM(i,2) = tmp(2) - mean(2);
		    }

		    //test if cluster fits to our expected pattern
		    Eigen::Matrix<double,3,3> covSum_ = planeM.transpose()*planeM;
		    Eigen::Matrix<double,3,3> cov_ = covSum_/(it->indices.size()-1);

		    //do PCA
		    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov_);
		    /*if(eig.eigenvalues()(0) < 0.00001 || eig.eigenvalues()(1) < 0.0001|| eig.eigenvalues()(2) < 0.001) {
			ROS_WARN("Cluster evals are not good");
			continue;
		    }*/
		    //check on eval shape
		    if( eig.eigenvalues()(0)/eig.eigenvalues()(2) > eval_thresh || eig.eigenvalues()(1)/eig.eigenvalues()(2) > eval_thresh) {
			ROS_WARN("Cluster is not cylindrical");
			continue;
		    }
		    //sanity check on up direction
		    if(eig.eigenvectors().col(2).dot(Eigen::Vector3d::UnitX()) > eig.eigenvectors().col(2).dot(Eigen::Vector3d::UnitZ()) ||
			    eig.eigenvectors().col(2).dot(Eigen::Vector3d::UnitY()) > eig.eigenvectors().col(2).dot(Eigen::Vector3d::UnitZ()) ) {
			ROS_WARN("Cluster is not oriented along Z");
			continue;
		    }
			
		    //std::cout<<"Cluster cov: "<<cov_<<std::endl;
		    //std::cout<<"evals: "<<eig.eigenvalues()<<std::endl;
		    std::cout<<"evecs: "<<eig.eigenvectors()<<std::endl;	
		    
		    std::cout << "PointCloud representing the Cluster: " << it->indices.size () << " data points." << std::endl;
		    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit) {
			pt.x = cloud->points[*pit].x;
			pt.y = cloud->points[*pit].y;
			pt.z = cloud->points[*pit].z;
			resultCloud.push_back(pt);
		    }
		    //////
		    top_plane.g_data.push_back(-0.30);
		    inner_cylinder.g_type = hqp_controllers_msgs::TaskGeometry::CYLINDER;
		    outer_cylinder.g_type = hqp_controllers_msgs::TaskGeometry::CYLINDER;
		    
		    inner_cylinder.g_data.push_back(mean(0));
		    inner_cylinder.g_data.push_back(mean(1));
		    inner_cylinder.g_data.push_back(mean(2));
		    outer_cylinder.g_data.push_back(mean(0));
		    outer_cylinder.g_data.push_back(mean(1));
		    outer_cylinder.g_data.push_back(mean(2));
		    
		    inner_cylinder.g_data.push_back(normal(0));
		    inner_cylinder.g_data.push_back(normal(1));
		    inner_cylinder.g_data.push_back(normal(2));
		    outer_cylinder.g_data.push_back(normal(0));
		    outer_cylinder.g_data.push_back(normal(1));
		    outer_cylinder.g_data.push_back(normal(2));
		    
		    inner_cylinder.g_data.push_back(3*sqrt(eig.eigenvalues()(1)) + grow_cylinder_m);
		    outer_cylinder.g_data.push_back(3*sqrt(eig.eigenvalues()(1)) + grow_cylinder_m + inner2outer);

		    res.CanTask.push_back(bottom_plane);
		    res.CanTask.push_back(top_plane);
		    res.CanTask.push_back(inner_cylinder);
		    res.CanTask.push_back(outer_cylinder);

		    res.success = true;
		    res.reference_frame = world_frame;
		    std::cerr<<"found can: "<<normal.transpose()<<" "<<mean.transpose()<<std::endl;
		    break;
		    /////
		    j++;
		}
#else

		hqp_controllers_msgs::TaskGeometry contact_point, approach_vector;
		contact_point.g_type = hqp_controllers_msgs::TaskGeometry::POINT;
		approach_vector.g_type = hqp_controllers_msgs::TaskGeometry::LINE;
		std::vector<pcl::PointIndices> cluster_indices;
		pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
		ec.setClusterTolerance (0.04); // 2cm
		ec.setMinClusterSize (500);
		ec.setMaxClusterSize (55000);
		ec.setSearchMethod (tree);
		ec.setInputCloud (cloud);
		ec.extract (cluster_indices);

		//find largest cluster within reach
		std::vector<pcl::PointIndices>::const_iterator jt;
		size_t n_pts = 0;
		int j=0;
		for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
		{
		    std::cout << "PointCloud representing the Cluster: " << it->indices.size() << " data points." << std::endl;
		    if(n_pts > it->indices.size()) {
			ROS_WARN("small cluster, skipping. npts is %lu",n_pts);
			continue;
		    } else {
			    std::cout << "Check Cluster: " << it->indices.size () << " data points." << std::endl;

		    }
		    Eigen::MatrixXd planeM (it->indices.size (),3);
		    Eigen::Vector3d mean;
		    mean<<0,0,0;
		    uint8_t r = 0, g = 55 + ((double)j/cluster_indices.size())*200., b = 0; // Example: Red color
		    j++;
		    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
		    float col = *reinterpret_cast<float*>(&rgb);
		    pcl::PointXYZRGB pt;
		    pt.rgb = col;
		    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit) {
			pt.x = cloud->points[*pit].x;
			pt.y = cloud->points[*pit].y;
			pt.z = cloud->points[*pit].z;
			Eigen::Vector3d tmp;
			tmp<<pt.x,pt.y,pt.z;
			mean += tmp;
		    }
		    mean /= (it->indices.size ());
		  
		    std::cerr<<"Found cluster at "<<mean.transpose()<<std::endl; 
		    if(mean(0) > max_x) {
		       ROS_WARN("Ignoring far away cluster\n");
		       continue;
		    }
		    n_pts = it->indices.size();
		    jt = it;
		    //For display...
		    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit) {
			pt.x = cloud->points[*pit].x;
			pt.y = cloud->points[*pit].y;
			pt.z = cloud->points[*pit].z;
			resultCloud.push_back(pt);
		    }
		}

		if(n_pts > 0) {
		    ROS_INFO("Found largest cluster with %lu points",n_pts);
		    double min_dist = INT_MAX;
		    Eigen::Vector3d closest_point;
		    uint8_t r = 200, g = 25, b = 20; // Example: Red color
		    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
		    float col = *reinterpret_cast<float*>(&rgb);
		    pcl::PointXYZRGB pt;
		    pt.rgb = col;
		    Eigen::Vector3d mean;
		    for (std::vector<int>::const_iterator pit = jt->indices.begin (); pit != jt->indices.end (); ++pit) {
			pt.x = cloud->points[*pit].x;
			pt.y = cloud->points[*pit].y;
			pt.z = cloud->points[*pit].z;
			Eigen::Vector3d tmp;
			tmp<<pt.x,pt.y,pt.z;
			mean += tmp;
			if((tmp-world2palm.translation()).norm() < min_dist) {
			    min_dist = (tmp-world2palm.translation()).norm();
			    closest_point = tmp;
			}
		    }
		    mean /= (jt->indices.size ());
	            std::cerr<<"Gripper is at "<<world2palm.translation().transpose()<<std::endl;
		    closest_point(2) = 0.26;
		    pt.x = closest_point(0);
		    pt.y = closest_point(1);
		    pt.z = closest_point(2);
		    resultCloud.push_back(pt);
		    //approach direction always points towards the the closest point
		    //Eigen::Vector3d approach = closest_point - world2palm.translation();
		    Eigen::Vector3d approach = mean - world2palm.translation();
		    approach.normalize();
		    closest_point = closest_point - dist_factor*approach; 
		    closest_point(2) = 0.26;
		    pt.x = closest_point(0);
		    pt.y = closest_point(1);
		    pt.z = closest_point(2);
		    resultCloud.push_back(pt);
		    /////
		    contact_point.g_data.push_back(closest_point(0));
		    contact_point.g_data.push_back(closest_point(1));
		    contact_point.g_data.push_back(closest_point(2));
		    approach_vector.g_data.push_back(closest_point(0));
		    approach_vector.g_data.push_back(closest_point(1));
		    approach_vector.g_data.push_back(closest_point(2));
		    approach_vector.g_data.push_back(approach(0));
		    approach_vector.g_data.push_back(approach(1));
		    approach_vector.g_data.push_back(approach(2));
		    res.CanTask.push_back(contact_point);
		    res.CanTask.push_back(approach_vector);
		    res.success = true;
		    res.reference_frame = world_frame;
		    std::cerr<<"found can: "<<closest_point.transpose()<<" "<<approach.transpose()<<std::endl;
		}
		    		    
		
#endif	
	    }

	    resultCloud.is_dense = false;
	    resultCloud.width = resultCloud.points.size();
	    resultCloud.height=1;
	    sensor_msgs::PointCloud2 rpub;
	    pcl::toROSMsg(resultCloud,rpub);
	    rpub.header.frame_id = world_frame;
	    rpub.header.stamp = ros::Time::now();
	    pcloud_pub_.publish(rpub);

	    return true;
	}

};


int main( int argc, char* argv[] )
{
    ros::init(argc, argv, "can_finder");
    CanFinderNode canNode;
   // ros::AsyncSpinner spinner(4); // Use 4 threads
   // spinner.start();
   // ros::waitForShutdown();

    ros::spin();
    return 0;
}

