#ifndef GEOMETRIC_CONSTRAINT_HH
#define GEOMETRIC_CONSTRAINT_HH

#include <Eigen/Core>

/** Different classes that implement various geometric primitives. 
  They all have an overloaded operator () over an Eigen vector. 
  The operator returns true if the constraint is satisfied.
  */

//TODO: Base class for constraints and derive

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

#endif
