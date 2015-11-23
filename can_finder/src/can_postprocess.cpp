#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>

#include<pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <tf/transform_listener.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>

#include <Eigen/Core>

class CanPPNode {

    private:
	ros::NodeHandle nh_;
	ros::NodeHandle n_;

	ros::Subscriber pcloud_sub;
        //tf::TransformListener tl;

	pcl::PointCloud<pcl::PointXYZRGB> my_cloud;
	std::string pcloud_topic;
	std::string pcloud_frame_name;
	std::string world_frame;
	std::string palm_frame;
	Eigen::Affine3d world2palm;

    public:

	CanPPNode() {
	    nh_ = ros::NodeHandle("~");
	    n_ = ros::NodeHandle();
	    nh_.param<std::string>("pcloud_topic", pcloud_topic,"result_cloud");
	    nh_.param<std::string>("world_frame", world_frame,"world");
	    nh_.param<std::string>("palm_frame", palm_frame,"velvet_fingers_palm");
	    
	    pcloud_frame_name = "";
	    pcloud_sub = n_.subscribe(pcloud_topic, 1, &CanPPNode::cloudCallback, this);

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
	    pcl::PointCloud<pcl::PointXYZRGB> points;
	    
	    uint8_t r0 = 0, g0 = 55, b0 = 0; // Example: Red color

	    for(int i=0; i<my_cloud.points.size(); ++i) {
		uint32_t rgb = *reinterpret_cast<int*>(&my_cloud.points[i].rgb);
		uint8_t r = (rgb >> 16) & 0x0000ff;
		uint8_t g = (rgb >> 8)  & 0x0000ff;
		uint8_t b = (rgb)       & 0x0000ff;
		if( r == r0 && g == g0 && b == b0 ) {
		    points.points.push_back(my_cloud.points[i]);
		}
	    }
	    //std::cout<<std::endl;
	   
	    if(points.points.size()<6){
		points.clear();
		ROS_ERROR("There were no clusters in this point cloud!\n");
		return;
	    }

	    Eigen::Vector3d mean_;
	    mean_<<0,0,0;
	    for(unsigned int i=0; i< points.size(); i++)
	    {
		Eigen::Vector3d tmp;
		tmp<<points[i].x,points[i].y,points[i].z;
		mean_ += tmp;
	    }
	    mean_ /= (points.size());
	    Eigen::MatrixXd mp;
	    mp.resize(points.size(),3);
	    for(unsigned int i=0; i< points.size(); i++)
	    {
		mp(i,0) = points[i].x - mean_(0);
		mp(i,1) = points[i].y - mean_(1);
		mp(i,2) = points[i].z - mean_(2);
	    }
	    Eigen::Matrix3d cov_ = mp.transpose()*mp/(points.size()-1);

	    std::cout<<"Mean: "<<mean_<<std::endl;
	    std::cout<<"Cov: "<<cov_<<std::endl;

	    /*
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
	    */
	
	}

};


int main( int argc, char* argv[] )
{
    ros::init(argc, argv, "can_pp");
    CanPPNode canNode;
   // ros::AsyncSpinner spinner(4); // Use 4 threads
   // spinner.start();
   // ros::waitForShutdown();

    ros::spin();
    return 0;
}

