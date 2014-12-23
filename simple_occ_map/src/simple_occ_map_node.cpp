#include <simple_occ_map/SimpleOccMap.hh>
#include <simple_occ_map/ConstraintMap.hh>
#include <simple_occ_map/SimpleOccMapMsg.h>

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
	ConstraintMap *map;
	ConstraintMap *object_map;
	ConstraintMap loaded;
	SimpleOccNode():resolution(0.01),map_size(2) {

	    nh_ = ros::NodeHandle("~");
	    n_ = ros::NodeHandle();

	    map_publisher_ = nh_.advertise<simple_occ_map::SimpleOccMapMsg> ("map_topic",10);
	    object_map_publisher_ = nh_.advertise<simple_occ_map::SimpleOccMapMsg> ("object_map_topic",10);
	    map = new ConstraintMap();
	    nh_.param<std::string>("gripper_file",fname,"test_small.cons");
	    map->loadGripperConstraints(fname.c_str());
	    object_map = new ConstraintMap(0,0,0,resolution,map_size/resolution,map_size/resolution,map_size/resolution);
	}
	~SimpleOccNode() {
	    delete map;
	    delete object_map;
	}

	void publishMap() {
	    simple_occ_map::SimpleOccMapMsg msg;
	    object_map->toMessage(msg);
	    object_map_publisher_.publish(msg);
	    
	    simple_occ_map::SimpleOccMapMsg msg2;
	    map->toMessage(msg2);
	    map_publisher_.publish(msg2);
	    //ROS_INFO("Publishing map");

	}

	void addScene() {
	    ROS_INFO("set some random cell to occ");
	    
	    Eigen::Affine3f pose;
	    pose.setIdentity();
	    object_map->drawCylinder(pose,0.1,0.5);

	    //pose = Eigen::AngleAxisf((float)rand()/RAND_MAX,Eigen::Vector3f::UnitX()) *
	    //	Eigen::AngleAxisf((float)rand()/RAND_MAX,Eigen::Vector3f::UnitY()) *Eigen::AngleAxisf((float)rand()/RAND_MAX,Eigen::Vector3f::UnitZ());
	    pose.translation()<<0.1,0.1,0;
	    Eigen::Vector3f box_size(0.1, 0.2, 0.3);
	    object_map->drawBox(pose,box_size);
	    
	    pose.translation()<<-0.4,-0.35,0;
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
    CylinderConstraint cc(pose,0.115,0.5);
    nd.map->computeValidConfigs(nd.object_map, cc);
    
    nd.map->resetMap(); 
    nd.map->drawValidConfigsSmall(); 
    nd.publishMap();
    ros::spinOnce();

    while(ros::ok()) {
	//sleep(20);
	//nd.publishMap();
	ros::spinOnce();
	loop_rate.sleep();
    }

    return 0;
}
