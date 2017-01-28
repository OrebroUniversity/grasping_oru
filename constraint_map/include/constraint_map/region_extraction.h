#ifndef _REGION_EXTRACTION_HH
#define _REGION_EXTRACTION_HH

#include <constraint_map/simple_occ_map.h>

/** @brief stores the bottom left and upper right indeces of a 3D cube.
  */
class CellIdxCube {
    public:
	CellIndex bl, ur;
	int volume() const{
	   return (((ur.i-bl.i)+1)*((ur.j-bl.j)+1)*((ur.k-bl.k)+1)); 
	}
};

/** @brief comparison operator for ranking cubes by volume */
inline bool cubecmpr(const CellIdxCube &a, const CellIdxCube &b) {
    return a.volume() > b.volume();
}

/** @brief Interface for region extractors. Must implement method for finding the largest empty region */
class EmptyRegionExtractor {

    public: 
	virtual CellIdxCube getMaxCube(SimpleOccMap *map) = 0;

};

#endif
