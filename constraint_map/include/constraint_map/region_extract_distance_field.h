#ifndef REGION_EXTRACT_DISTANCE_FIELD
#define REGION_EXTRACT_DISTANCE_FIELD

#include <constraint_map/region_extraction.h>

//typedef int Triplet [3];
class Triplet {
    public:
	int data[3];
	Triplet(){};
	Triplet(int i, int j, int k) {
	    data[0]=i, data[1]=j, data[2]=k;
	}
	~Triplet() {};
	Triplet(const Triplet &other) {
	    memcpy(&data,&other.data,3*sizeof(int));
	}
	inline int& operator[](int i) {
	    return data[i];
	}
};

inline int minval (int a, int b) {
	return a < b ? a : b;
}

/** @brief a 3D Distance grid where each cell stores distance to closest occupied cell along each of the three dimensions */
class DistanceGrid {
    public:
    bool loopX, loopY, loopZ;
    int size_x,size_y,size_z;
    bool isAlloc;
    Triplet ***distance_grid;
    DistanceGrid() { isAlloc=false; }
    DistanceGrid(int sx, int sy, int sz);
    ~DistanceGrid();
    Triplet at(int i, int j, int k);
    Triplet at(CellIndex id) {
	return this->at(id.i, id.j, id.k);
    }

    //computes the distance grid
    void computeDistanceGrid(SimpleOccMap *map);
};

/** @brief Extractor based on the 3D distance field */
class DfunMaxEmptyCubeExtractor : public EmptyRegionExtractor {
    public:
	std::vector<CellIdxCube> empty_cubes;
	bool loopX, loopY, loopZ;
	DfunMaxEmptyCubeExtractor():loopX(false),loopY(false),loopZ(false) {};

	//search for rectangle
	virtual CellIdxCube getMaxCube(SimpleOccMap *map);
};

#endif
