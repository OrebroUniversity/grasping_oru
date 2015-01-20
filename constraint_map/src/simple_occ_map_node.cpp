#include <constraint_map/SimpleOccMap.hh>
#include <constraint_map/ConstraintMap.hh>
#include <constraint_map/SimpleOccMapMsg.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <ros/ros.h>

class SimpleOccNode {

    private:
	// Our NodeHandle, points to home
	ros::NodeHandle nh_;
	//global node handle
	ros::NodeHandle n_;

	ros::Publisher map_publisher_;
	ros::Publisher object_map_publisher_;

	float map_size;
	float resolution;
	std::string fname;

    public:
	tf::TransformBroadcaster br;
	ConstraintMap *map;
	ConstraintMap *object_map;
	ConstraintMap loaded;
	SimpleOccNode():resolution(0.01),map_size(2) {

	    nh_ = ros::NodeHandle("~");
	    n_ = ros::NodeHandle();

	    map_publisher_ = nh_.advertise<constraint_map::SimpleOccMapMsg> ("map_topic",10);
	    object_map_publisher_ = nh_.advertise<constraint_map::SimpleOccMapMsg> ("object_map_topic",10);
	    map = new ConstraintMap();
	    //nh_.param<std::string>("gripper_file",fname,"test_small.cons");
	    nh_.param<std::string>("gripper_file",fname,"full.cons");
	    map->loadGripperConstraints(fname.c_str());
	    object_map = new ConstraintMap(0,0,0,resolution,map_size/resolution,map_size/resolution,map_size/resolution);
	}
	~SimpleOccNode() {
	    delete map;
	    delete object_map;
	}

	void publishMap() {
	    constraint_map::SimpleOccMapMsg msg;
	    object_map->toMessage(msg);
	    msg.header.frame_id = "/map_frame";
	    object_map_publisher_.publish(msg);
	    
	    constraint_map::SimpleOccMapMsg msg2;
	    map->toMessage(msg2);
	    msg2.header.frame_id = "/gripper_frame";
	    map_publisher_.publish(msg2);
	    ROS_INFO("Publishing map");

	}

	void addScene() {
	    ROS_INFO("set some random cell to occ");
	    
	    Eigen::Affine3f pose;
	    pose.setIdentity();
	    pose = Eigen::AngleAxisf(-0.57,Eigen::Vector3f::UnitX());
	    //pose.translation()<<-0.25,-0.1,-0.1;
	    object_map->drawCylinder(pose,0.1,0.5);

	    //pose = Eigen::AngleAxisf((float)rand()/RAND_MAX,Eigen::Vector3f::UnitX()) *
	    //	Eigen::AngleAxisf((float)rand()/RAND_MAX,Eigen::Vector3f::UnitY()) *Eigen::AngleAxisf((float)rand()/RAND_MAX,Eigen::Vector3f::UnitZ());
	    pose.setIdentity();
	    pose.translation()<<-0.25,-0.1,-0.1;
	    Eigen::Vector3f box_size(0.1, 0.2, 0.8);
	    object_map->drawBox(pose,box_size);
	    
	    pose.translation()<<-0.5,-0.5,-0.1;
	    box_size <<1, 1, 0.02;
	    object_map->drawBox(pose,box_size);
	    
	    pose.translation()<<0.1,-0.35,-0.1;
	    box_size <<0.2, 0.2, 0.4;
	    object_map->drawBox(pose,box_size);
	    
	    pose = Eigen::AngleAxisf((float)rand()/RAND_MAX,Eigen::Vector3f::UnitX()) *
		Eigen::AngleAxisf((float)rand()/RAND_MAX,Eigen::Vector3f::UnitY()) *Eigen::AngleAxisf((float)rand()/RAND_MAX,Eigen::Vector3f::UnitZ());
	    pose.translation()<<(float)rand()/RAND_MAX,(float)rand()/RAND_MAX,(float)rand()/RAND_MAX;
	    pose.translation() = map_size/4*pose.translation();
	    //object_map->drawBox(pose,box_size);

	    object_map->updateMap();
	}


};

int main(int argc, char **argv) {

    ros::init(argc,argv,"simple_node");
    SimpleOccNode nd;
    ros::Rate loop_rate(0.5);
    //ros::Rate loop_rate(500);
    //nd.map->sampleGripperGrid(3, 20, 10, 0.1, 0.5, 0.1, 0.6);

    nd.addScene();
    Eigen::Affine3f pose;
    pose.setIdentity();
    pose = Eigen::AngleAxisf(-0.57,Eigen::Vector3f::UnitX());
    //pose.translation()<<-0.25,-0.1,-0.1;
    CylinderConstraint cc(pose,0.115,0.5);
    nd.map->computeValidConfigs(nd.object_map, cc);

    tf::Transform transform;
    tf::transformEigenToTF(pose.cast<double>(),transform);
    nd.br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map_frame", "gripper_frame"));

    
    nd.map->resetMap(); 
    nd.map->drawValidConfigsSmall(); 
    nd.publishMap();
    ros::spinOnce();

    while(ros::ok()) {
	    nd.br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map_frame", "gripper_frame"));
	//sleep(2);
	//nd.publishMap();
	ros::spinOnce();
	loop_rate.sleep();
    }

    return 0;
}
