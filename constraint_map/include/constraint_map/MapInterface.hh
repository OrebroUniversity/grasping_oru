#ifndef __SIMPLE_OCC_MAP_INTERFACE_HH
#define __SIMPLE_OCC_MAP_INTERFACE_HH

#include <constraint_map/SimpleOccMapMsg.h>

struct CellIndex {
    int i;
    int j;
    int k;
};

class SimpleOccMapIfce {
    public:
	virtual void toMessage(constraint_map::SimpleOccMapMsg &msg) = 0;
	//virtual void getIntersection(const SimpleOccMapIfce *other, std::vector<CellIndex> &idx_this) const = 0;
	//virtual void getIntersectionWithPose(const SimpleOccMapIfce *other, Eigen::Affine3f &this_to_map, std::vector<CellIndex> &idx_this) const = 0;
	virtual bool isOccupied(const Eigen::Vector3f &point) const = 0;
};

#endif
