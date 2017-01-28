#ifndef CONSTRAINT_MAP_HH
#define CONSTRAINT_MAP_HH

#include <constraint_map/simple_occ_map.h>
#include <constraint_map/geometric_constraint.h>
#include <constraint_map/gripper_model.h>
#include <constraint_map/gripper_configuration.h>
#include <constraint_map/region_extraction.h>
#include <constraint_map/region_extract_distance_field.h>
#include <constraint_map/region_extract_segment_tree.h>

#include <sys/time.h>
#include <cstdio>

#define _FILE_VERSION_IROS2016 "#F V0.2"
#define _FILE_VERSION_ "#F V0.3"

class ConstraintMap : public SimpleOccMap {

    using SimpleOccMap::center;
    using SimpleOccMap::size_meters;
    using SimpleOccMap::resolution;
    using SimpleOccMap::size_x;
    using SimpleOccMap::size_y;
    using SimpleOccMap::size_z;
    using SimpleOccMap::grid;
    using SimpleOccMap::isInitialized;

    private: 
	std::vector<BoxConstraint> boxes;
	std::vector<CylinderConstraint> cylinders;
	
	GripperModel *model;
	GripperModelPJ *modelPJ;
	SimpleOccMap *config_sample_grid;
	bool hasGripper;
	bool isSphereGrid;
	bool isPJGripper;

	int n_v,n_o,n_d;
	float min_z, max_z, min_dist, max_dist;	
	std::vector<GripperConfiguration*> valid_configs;
	std::vector<bool> palmCollision, fingerCollision, emptyGripper, orientationFilter;
	ConfigurationList *** config_grid;

	double getDoubleTime()
	{
	    struct timeval time;
	    gettimeofday(&time,NULL);
	    return time.tv_sec + time.tv_usec * 1e-6;
	}
	CellIdxCube cube;

    public:
	ConstraintMap():SimpleOccMap() { hasGripper = false; config_sample_grid = NULL; isSphereGrid=false; isPJGripper=false;};
	ConstraintMap(float _cen_x, float _cen_y, float _cen_z, 
		float _resolution, int _size_x, int _size_y, int _size_z, bool isVelvet=true):SimpleOccMap(_cen_x,_cen_y,_cen_z,_resolution,_size_x,_size_y,_size_z) { 

	    if(isVelvet) {
		//FIXME: these should be read in somewhere
		Eigen::Vector3f finger_size(0.06,0.129,0.08);
		Eigen::Vector3f palm_size(0.18,0.13,0.1);
		Eigen::Affine3f palm2left;
		palm2left.setIdentity();
		palm2left.translation() = Eigen::Vector3f(-0.025,0.1,0);
		Eigen::Affine3f palm2right;
		palm2right.setIdentity();
		palm2right.translation() = Eigen::Vector3f(0.025,0.1,0);
		Eigen::Affine3f palm2fingers;
		palm2fingers.setIdentity();
		palm2fingers.translation() = Eigen::Vector3f(0.0,0.1,0);
		model = new GripperModel(finger_size,palm_size,palm2left,palm2right,palm2fingers);
		modelPJ = NULL;
		isPJGripper = false;
	    } else {
		model = NULL;
		isPJGripper = true;
		
		Eigen::Vector3f finger_size(0.01,0.04,0.015);
		Eigen::Vector3f palm_size(0.08,0.085,0.068);
		Eigen::Affine3f palm2fingers;
		palm2fingers.setIdentity();
		palm2fingers.translation() = Eigen::Vector3f(0.0,0.09,0);
		modelPJ = new GripperModelPJ(finger_size,palm_size,palm2fingers,0.05);

	    }	
	    hasGripper = true; 
	    isSphereGrid=false; 
	    initializeConfigs();
	    config_sample_grid = NULL;
	} ;

	virtual ~ConstraintMap() {
	    if(hasGripper) delete model;
	    if(isInitialized && config_grid != NULL) {
		for (int i=0; i<size_x; ++i) {
		    for (int j=0; j<size_y; ++j) {
			delete[] config_grid[i][j];
		    }
		    delete[] config_grid[i];
		}
		delete[] config_grid;
	    }
	    for(int i=0; i<valid_configs.size(); ++i) {
		if(valid_configs[i]!=NULL) delete valid_configs[i];
	    }
	}

	void initializeConfigs() {
	    if(resolution < 0) return;
	    config_grid = new ConfigurationList** [size_x];
	    for (int i=0; i<size_x; ++i) {
		config_grid[i] = new ConfigurationList* [size_y];
		for (int j=0; j<size_y; ++j) {
		    config_grid[i][j] = new ConfigurationList[size_z];
		}
	    }
	}

	//high-level drawing functions
	void drawBox (Eigen::Affine3f pose, Eigen::Vector3f size) {
	    BoxConstraint bc(pose,size);
	    boxes.push_back(bc);
	}
	void drawCylinder (Eigen::Affine3f pose, float radius, float height) {
	    CylinderConstraint cc(pose,radius,height);
	    cylinders.push_back(cc);
	}
	void drawGripper( GripperConfigurationSimple &config) {
	    if(hasGripper) {
		config.model = model;
		config.calculateConstraints();
		boxes.push_back(config.palm);
		boxes.push_back(config.leftFinger);
		boxes.push_back(config.rightFinger);
	    }
	}

	void drawValidConfigs();
	void drawValidConfigsSmall();
	void generateOpeningAngleDump(std::string &fname);

	//returns a colored point cloud of configs. red are invalid, orange valid, green valid and selected
	void getConfigsForDisplay(pcl::PointCloud<pcl::PointXYZRGB> &configs);

	void sampleGripperGrid(int n_vert_slices, int n_orient, int n_dist_samples,
		float min_z, float max_z, float min_dist, float max_dist);
	
	void sampleGripperGridSphere(int n_vert_angles, int n_orient_angles, int n_dist_samples,
		float min_dist, float max_dist);

	void updateMap();
	void updateMapAndGripperLookup();

	//computes the valid gripper configurations when grasping a cylinder inside object map
	void computeValidConfigs(SimpleOccMapIfce *object_map, Eigen::Affine3f cpose, float cradius, float cheight, 
		    Eigen::Vector3f &prototype_orientation, float orientation_tolerance, GripperPoseConstraint &output);
	
	bool saveGripperConstraints(const char *fname) const;
	bool loadGripperConstraints(const char *fname);

    private:
	void saveMatrix4f(Eigen::Matrix4f &m, FILE *fout) const {
	    const float *data = m.data();
	    int size = m.rows()*m.cols();
	    fwrite(data, sizeof(float), size, fout);
	}
	bool readMatrix4f(Eigen::Matrix4f &m, FILE *fin) {
	    int size = m.rows()*m.cols();
	    if(fread(m.data(), sizeof(float), size, fin)!=size) {
		std::cerr<<"couldn't read matrix\n";
		return false;
	    }
	    return true;
	}
	bool getPoseForConfig(CellIndex &config_index, Eigen::Affine3f &pose);

    public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif
