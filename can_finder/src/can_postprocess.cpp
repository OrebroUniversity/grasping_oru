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

#include <sensor_msgs/Image.h>
#include <fstream>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Core>

class CanPPNode {

    private:
	ros::NodeHandle nh_;
	ros::NodeHandle n_;

	ros::Subscriber pcloud_sub;
	ros::Subscriber img_sub;

	pcl::PointCloud<pcl::PointXYZRGB> my_cloud;
	std::string pcloud_topic;
	std::string pcloud_frame_name;
	std::string outfile_name;
	std::string image_topic;
	std::ofstream outfile;
	int ctr;
	Eigen::Affine3d world2palm;

    public:

	CanPPNode() {
	    nh_ = ros::NodeHandle("~");
	    n_ = ros::NodeHandle();
	    nh_.param<std::string>("pcloud_topic", pcloud_topic,"result_cloud");
	    nh_.param<std::string>("out", outfile_name,"volumes.m");
	    nh_.param<std::string>("image_topic", image_topic,"rgb_image_raw");
	    
	    pcloud_sub = n_.subscribe(pcloud_topic, 1, &CanPPNode::cloudCallback, this);
	    img_sub = n_.subscribe(image_topic, 10, &CanPPNode::imageCallback, this);
	    outfile.open(outfile_name.c_str(), std::ofstream::out);
	    outfile<<"volumes = [";
	    ctr = 0;

	}
	
	~CanPPNode() {
	    outfile<<"];\n";
	    outfile.flush();
	    outfile.close();
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

	void imageCallback(const sensor_msgs::ImageConstPtr &msg) {
	    cv_bridge::CvImagePtr cv_ptr;
	    try
	    {
		cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	    }
	    catch (cv_bridge::Exception& e)
	    {
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	    }
	    cv::Mat dst;
	    cv::flip(cv_ptr->image, dst, -1);
	    
	    char fname[500];
	    snprintf(fname,499,"img%03d.jpg", ctr);
	    cv::imwrite(fname, dst);
	    ctr++;
	}

	void cloudCallback(const sensor_msgs::PointCloud2 &msg) {
	    
	    pcl::fromROSMsg(msg, my_cloud);
	    pcl::PointCloud<pcl::PointXYZRGB> points;
	    
	    uint8_t r0 = 0, g0 = 250, b0 = 0; // Example: Red color

	    for(int i=0; i<my_cloud.points.size(); ++i) {
		uint32_t rgb = *reinterpret_cast<int*>(&my_cloud.points[i].rgb);
		uint8_t r = (rgb >> 16) & 0x0000ff;
		uint8_t g = (rgb >> 8)  & 0x0000ff;
		uint8_t b = (rgb)       & 0x0000ff;
		if( r == r0 && g == g0 && b == b0 ) {
		    points.points.push_back(my_cloud.points[i]);
		}
	    }

	    std::cout<<"Got configs: "<<my_cloud.points.size()<<" selected "<<points.size()<<std::endl;
	    outfile<<points.size()<<" ";
	    outfile.flush();
	
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

