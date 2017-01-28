#ifndef GRIPPER_CONFIGURATION_HH
#define GRIPPER_CONFIGURATION_HH

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

#endif
