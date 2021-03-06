#ifndef CONSTRAINT_MAP_HH
#define CONSTRAINT_MAP_HH

#include <constraint_map/SimpleOccMap.hh>
#include <constraint_map/RegionExtraction.hh>
#include <sys/time.h>
#include <cstdio>

class PlaneConstraint {
    public:
	Eigen::Vector3f a;
	float b;
	PlaneConstraint() {
	    a.setZero();
	    b = 0;
	}
	inline bool operator()(Eigen::Vector3f& x) {
	    return (a.dot(x) - b > 0);
	}
};

class BoxConstraint {
    public:
	Eigen::Matrix<float,6,3> A;
	Eigen::Matrix<float,6,1> b;
	BoxConstraint() { };
	BoxConstraint (Eigen::Affine3f &pose, Eigen::Vector3f size) {
	    calculateConstraints(pose,size);
	}
	void calculateConstraints(Eigen::Affine3f &pose, Eigen::Vector3f size) {
	    A.block<1,3>(0,0) = pose.rotation()*Eigen::Vector3f::UnitX();
	    A.block<1,3>(1,0) = -pose.rotation()*Eigen::Vector3f::UnitX();
	    A.block<1,3>(2,0) = pose.rotation()*Eigen::Vector3f::UnitY();
	    A.block<1,3>(3,0) = -pose.rotation()*Eigen::Vector3f::UnitY();
	    A.block<1,3>(4,0) = pose.rotation()*Eigen::Vector3f::UnitZ();
	    A.block<1,3>(5,0) = -pose.rotation()*Eigen::Vector3f::UnitZ();
	   
	    Eigen::Vector3f tr = pose.translation();
	    b(0) = tr.dot(A.block<1,3>(0,0));
	    b(1) = tr.dot(A.block<1,3>(1,0))-size(0);
	    b(2) = tr.dot(A.block<1,3>(2,0));
	    b(3) = tr.dot(A.block<1,3>(3,0))-size(1);
	    b(4) = tr.dot(A.block<1,3>(4,0));
	    b(5) = tr.dot(A.block<1,3>(5,0))-size(2);
	    //b(0) = tr(0);
	    //b(1) = -(tr(0)+size(0));
	    //b(2) = tr(1);
	    //b(3) = -(tr(1)+size(1));
	    //b(4) = tr(2);
	    //b(5) = -(tr(2)+size(2));
	}
	inline bool operator()(Eigen::Vector3f &x) {
	    Eigen::Matrix<float,6,1> bp = A*x-b;
	    if(bp(0)<0) return false;
	    if(bp(1)<0) return false;
	    if(bp(2)<0) return false;
	    if(bp(3)<0) return false;
	    if(bp(4)<0) return false;
	    if(bp(5)<0) return false;
	    return true;
	}
	
    public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

class CylinderConstraint {

    public:
	Eigen::Matrix<float,2,3> A;
	Eigen::Matrix<float,2,1> b;
	float radius_, height_;
	Eigen::Affine3f pose;
	CylinderConstraint() {
	    A.setZero();
	    b.setZero();
	    radius_ = 0;
	    height_ = 0;
	}
	CylinderConstraint(Eigen::Affine3f &pose_, float radius, float height) {
	    calculateConstraints(pose_,radius,height);
	}
	void calculateConstraints(Eigen::Affine3f &pose_, float radius, float height) {
	    pose = pose_;
	    A.block<1,3>(0,0) = pose.rotation()*Eigen::Vector3f::UnitZ();
	    A.block<1,3>(1,0) = -pose.rotation()*Eigen::Vector3f::UnitZ();
	    Eigen::Vector3f tr = pose.translation();
	    b(0) = tr.dot(A.block<1,3>(0,0));
	    b(1) = tr.dot(A.block<1,3>(1,0))-height;
	    radius_ = radius;
	    height_ = height;
	}
	float getHeight() {
	    return height_;
	    //return pose.translation().dot(A.block<1,3>(1,0)) - b(1);
	} 
	inline bool operator()(Eigen::Vector3f &x) {
	    Eigen::Matrix<float,2,1> bp = A*x-b;
	    if(bp(0)<0) return false;
	    if(bp(1)<0) return false;
	    Eigen::Vector3f normal = A.block<1,3>(0,0).transpose();
	    Eigen::Vector3f xt = x - pose.translation();
	    Eigen::Vector3f rejection = xt - xt.dot(normal)*normal;
	    if(rejection.norm() > radius_) return false;
	    return true;
	}
    public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class HalfCylinderConstraint {

    public:
	Eigen::Matrix<float,3,3> A;
	Eigen::Matrix<float,3,1> b;
	float radius_;
	Eigen::Affine3f pose;

	HalfCylinderConstraint() {};
	HalfCylinderConstraint(Eigen::Affine3f &pose_, float radius, float height, float slice_angle) {
	    calculateConstraints(pose_,radius,height,slice_angle);
	}
	void calculateConstraints(Eigen::Affine3f &pose_, float radius, float height, float slice_angle) {
	    pose = pose_;
	    A.block<1,3>(0,0) = pose.rotation()*Eigen::Vector3f::UnitZ();
	    A.block<1,3>(1,0) = -pose.rotation()*Eigen::Vector3f::UnitZ();
	    A.block<1,3>(2,0) = pose.rotation()*Eigen::AngleAxisf(slice_angle,Eigen::Vector3f::UnitZ())*Eigen::Vector3f::UnitY();
	    Eigen::Vector3f tr = pose.translation();
	    b(0) = tr.dot(A.block<1,3>(0,0));
	    b(1) = tr.dot(A.block<1,3>(1,0))-height;
	    b(2) = tr.dot(A.block<1,3>(2,0));
	    radius_ = radius;
	}
	inline bool operator()(Eigen::Vector3f &x) {
	    Eigen::Matrix<float,3,1> bp = A*x-b;
	    if(bp(0)<0) return false;
	    if(bp(1)<0) return false;
	    if(bp(2)<0) return false;
	    Eigen::Vector3f normal = A.block<1,3>(0,0).transpose();
	    Eigen::Vector3f xt = x - pose.translation();
	    Eigen::Vector3f rejection = xt - xt.dot(normal)*normal;
	    if(rejection.norm() > radius_) return false;
	    return true;
	}
    public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class SphereConstraint {
    public:
	Eigen::Vector3f center;
	float radius;
	SphereConstraint() {};
	SphereConstraint(Eigen::Vector3f &c, float r) {
	    center=c;
	    radius=r;
	}
	inline bool operator()(Eigen::Vector3f &x) {
	    return (x-center).norm()<=radius;
	}
};

class GripperPoseConstraint {
    public:
	bool isSphere;
	CylinderConstraint inner_cylinder, outer_cylinder;
	SphereConstraint inner_sphere, outer_sphere;
	PlaneConstraint upper_plane, lower_plane, left_bound_plane, right_bound_plane;
	float cspace_volume, debug_time;
	float min_oa, max_oa;
};

///model for the velvet fingers RR gripper
class GripperModel {
    public:
	Eigen::Vector3f finger_size, palm_size;
	Eigen::Affine3f palm2left, palm2right, palm2fingers;
	Eigen::Affine3f palm2palm_box, right2right_box, left2left_box;
	float max_oa;
	float min_oa;
	GripperModel():min_oa(0),max_oa(M_PI) { };
	GripperModel(Eigen::Vector3f &finger_size_, Eigen::Vector3f &palm_size_,
		Eigen::Affine3f &palm2left_, Eigen::Affine3f &palm2right_, Eigen::Affine3f &palm2fingers_):min_oa(0),max_oa(M_PI) {
	    finger_size = finger_size_;
	    palm_size = palm_size_;
	    palm2left = palm2left_;
	    palm2right = palm2right_;
	    palm2fingers = palm2fingers_;
	    //now the offsets to the bottom left corner of each box
	    //assumption is that y points out of the gripper, x along finger opening
	    palm2palm_box.setIdentity();
	    palm2palm_box.translation() = -palm_size/2;
	    palm2palm_box.translation()(1) = 0;
	    left2left_box.setIdentity();
	    left2left_box.translation() =   -finger_size/2;
	    left2left_box.translation()(1) = 0;
	    right2right_box.setIdentity();
	    right2right_box.translation() = -finger_size/2;
	    right2right_box.translation()(1) = 0;

	}
    public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

///Parallel-jaw gripper
class GripperModelPJ {
    public:
	Eigen::Vector3f finger_size, palm_size;
	Eigen::Affine3f palm2palm_box;
	Eigen::Affine3f palm2fingers_box;
	float max_oa;
	float min_oa;
	float finger_thickness;
	GripperModelPJ():min_oa(0) { };
	GripperModelPJ(Eigen::Vector3f &finger_size_, Eigen::Vector3f &palm_size_, Eigen::Affine3f &palm2fingers_, float max_oa_):min_oa(0) {
	    finger_thickness = finger_size_(0);
	    finger_size = finger_size_;
	    max_oa = max_oa_;
	    //now let's calculate the "finger_size" for the finger sweep box: 2 finger widths + max_oa
	    finger_size(0) = 2*finger_size(0) + max_oa;
	    palm_size = palm_size_;
	    
	    //now the offsets to the bottom left corner of each box
	    //assumption is that y points out of the gripper, x along finger opening
	    palm2palm_box.setIdentity();
	    palm2palm_box.translation() = -palm_size/2;
	    palm2palm_box.translation()(1) = 0;
	    
	    Eigen::Affine3f f2f_box;
	    f2f_box.setIdentity();
	    f2f_box.translation() = -finger_size/2;
	    f2f_box.translation()(1) = 0;
	    palm2fingers_box = palm2fingers_*f2f_box;
	}
    public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class GripperConfigurationSimple {
    ///position of the gripper
    //Eigen::Vector3f position;
    ///vector pointing from the gripper to the center of the target
    //Eigen::Vector3f approach_direction;
    public:
	GripperConfigurationSimple() {};
	GripperConfigurationSimple(Eigen::Affine3f &pose_, float &oa, GripperModel *model_) {
	    pose = pose_;
	    opening_angle = oa;
	    model = model_;
	};
	GripperModel *model;
	Eigen::Affine3f pose;
	//opening angle of the gripper
	float opening_angle;
	BoxConstraint leftFinger, rightFinger, palm;

	void calculateConstraints() {
	    
	    Eigen::Affine3f ps;
	    ps = pose*model->palm2palm_box;
	    palm.calculateConstraints(ps,model->palm_size);
	      
	    Eigen::Affine3f opening_trans;
	    opening_trans = Eigen::AngleAxisf(opening_angle/2,Eigen::Vector3f::UnitZ());
	    ps = pose*model->palm2left*opening_trans*model->left2left_box;
	    leftFinger.calculateConstraints(ps,model->finger_size);
	    
	    opening_trans = Eigen::AngleAxisf(-opening_angle/2,Eigen::Vector3f::UnitZ());
	    ps = pose*model->palm2right*opening_trans*model->right2right_box;
	    rightFinger.calculateConstraints(ps,model->finger_size);
	}
    public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

class GripperConfiguration {
    protected:
	BoxConstraint palm_;
	HalfCylinderConstraint fingerSweep_;

    public:
	GripperConfiguration() {isValid = false;};
	GripperConfiguration(Eigen::Affine3f &pose_, GripperModel *model_) {
	    pose = pose_;
	    min_oa = 0; 
	    max_oa = M_PI; 
	    model = model_;
	    isValid = true;
	};

	GripperModel *model;
	Eigen::Affine3f pose, psm;
	//Eigen::Vector3f ori;
	//opening angle of the gripper
	float min_oa, max_oa;

	bool isValid;
	virtual bool palm(Eigen::Vector3f &x) {
	    return palm_(x);
	}

	virtual bool fingerSweep(Eigen::Vector3f &x) {
	    return fingerSweep_(x);
	}

	virtual void calculateConstraints() {
	    
	    Eigen::Affine3f ps;
	    ps = pose*model->palm2palm_box;
	    palm_.calculateConstraints(ps,model->palm_size);
	      
	    ps = pose*model->palm2fingers;
	    fingerSweep_.calculateConstraints(ps,model->finger_size(1),model->finger_size(2),0);
	    psm = ps;
	}

	virtual void updateMinAngle (Eigen::Vector3f &x) {
	    Eigen::Vector3f xt = psm.inverse()*x;
	    float mm = M_PI - 2*atan2f(xt(1),xt(0));
	    if(mm > min_oa) min_oa = mm;
	}
	virtual void updateMaxAngle (Eigen::Vector3f &x) {
	    Eigen::Vector3f xt = psm.inverse()*x;
	    xt(2) = 0; 
	    float mm = M_PI - 2*atan2f(xt(1),xt(0));
	    //now we need to remove a slice for the gripper thickness
	    float gt = acos(model->finger_size(0) / xt.norm());
	    mm = mm - gt;
	    mm = mm < 0 ? 0 : mm;
	    if( mm < max_oa) max_oa = mm;
	}
    public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

class GripperConfigurationPJ : public GripperConfiguration {
    protected:
	BoxConstraint fingerSweepPJ;

    public:
	GripperConfigurationPJ() {isValid = false;};
	GripperConfigurationPJ(Eigen::Affine3f &pose_, GripperModelPJ *model_) {
	    pose = pose_;
	    min_oa = 0.0; 
	    max_oa = 0.05; 
	    modelPJ = model_;
	    isValid = true;
	};

	GripperModelPJ *modelPJ;
	
	virtual bool fingerSweep(Eigen::Vector3f &x) {
	    return fingerSweepPJ(x);
	}

	virtual void calculateConstraints() {
	    
	    Eigen::Affine3f ps;
	    ps = pose*modelPJ->palm2palm_box;
	    palm_.calculateConstraints(ps,modelPJ->palm_size);
	      
	    ps = pose*modelPJ->palm2fingers_box;
	    fingerSweepPJ.calculateConstraints(ps,modelPJ->finger_size);
	    psm = ps;
	}

	virtual void updateMinAngle (Eigen::Vector3f &x) {
	    if(fingerSweep(x)) {
		Eigen::Vector3f xt = psm.inverse()*x;
		float mm = fabsf(xt(0));
		if(mm > min_oa) min_oa = mm;
	    }
	}
	virtual void updateMaxAngle (Eigen::Vector3f &x) {
	    if(fingerSweep(x)) {
		Eigen::Vector3f xt = psm.inverse()*x;
		float mm = fabsf(xt(0))-modelPJ->finger_thickness;
		mm = mm < 0 ? 0 : mm;
		if( mm < max_oa) max_oa = mm;
	    }
	}
    public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

struct ConfigurationList {
	std::vector<GripperConfiguration**> configs;
	std::vector<int> config_ids;
};

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
