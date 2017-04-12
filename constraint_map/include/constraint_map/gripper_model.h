#ifndef GRIPPER_MODEL_HH
#define GRIPPER_MODEL_HH

#include <constraint_map/geometric_constraint.h>

class GripperPoseConstraint {
    public:
	bool isSphere;
	CylinderConstraint inner_cylinder, outer_cylinder;
	SphereConstraint inner_sphere, outer_sphere;
	PlaneConstraint upper_plane, lower_plane, left_bound_plane, right_bound_plane;
	float cspace_volume, debug_time;
	float min_oa, max_oa;
};

//TODO: Specify a parent class and derive different gripper models from there

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

#endif
