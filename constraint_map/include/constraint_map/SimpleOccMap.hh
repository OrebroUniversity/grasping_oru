#ifndef SIMPLE_OCC_MAP_HH
#define SIMPLE_OCC_MAP_HH

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <constraint_map/MapInterface.hh>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class SimpleOccMap : public SimpleOccMapIfce {

    protected:
	Eigen::Vector3f center;
	Eigen::Vector3f size_meters;
	float resolution;

	bool isInitialized;

    public:
	int size_x, size_y, size_z;
	int ***grid;
	enum OccVal{ OCC=100, FREE=0, UNKN=-1};
	SimpleOccMap() { resolution = -1; isInitialized = false; };
	SimpleOccMap(float _cen_x, float _cen_y, float _cen_z, 
		float _resolution, int _size_x, int _size_y, int _size_z);
	SimpleOccMap(const SimpleOccMap& other);
	virtual ~SimpleOccMap();
	void initialize();

	inline float getResolution() const { return resolution; }

	///setters and getters per cell
	inline void setOccupied (int i, int j, int k) {
	    if(!isInitialized) return;
	    if(!isInside(i,j,k)) return;
	    grid[i][j][k] = SimpleOccMap::OCC;

	}
	inline void setFree (int i, int j, int k) {
	    if(!isInitialized) return;
	    if(!isInside(i,j,k)) return;
	    grid[i][j][k] = SimpleOccMap::FREE;

	}
	inline void setUnknown (int i, int j, int k) {
	    if(!isInitialized) return;
	    if(!isInside(i,j,k)) return;
	    grid[i][j][k] = SimpleOccMap::UNKN;

	}
	inline void setOccupied (const CellIndex &idx) {
	    if(!isInitialized) return;
	    if(!isInside(idx)) return;
	    grid[idx.i][idx.j][idx.k] = SimpleOccMap::OCC;

	}
	inline void setFree (const CellIndex &idx) {
	    if(!isInitialized) return;
	    if(!isInside(idx)) return;
	    grid[idx.i][idx.j][idx.k] = SimpleOccMap::FREE;

	}
	inline void setUnknown (const CellIndex &idx) {
	    if(!isInitialized) return;
	    if(!isInside(idx)) return;
	    grid[idx.i][idx.j][idx.k] = SimpleOccMap::UNKN;

	}

	///getters per cell
	inline bool isOccupied(int i, int j, int k) const {
	    if(!isInitialized) return false;
	    if(!isInside(i,j,k)) return false;
	    return (grid[i][j][k] == SimpleOccMap::OCC);

	}

	inline bool isFree(int i, int j, int k) const {
	    if(!isInitialized) return false;
	    if(!isInside(i,j,k)) return false;
	    return (grid[i][j][k] == SimpleOccMap::FREE);

	}

	inline bool isUnknown(int i, int j, int k) const {
	    if(!isInitialized) return false;
	    if(!isInside(i,j,k)) return false;
	    return (grid[i][j][k] == SimpleOccMap::UNKN);

	}

	inline bool isOccupied(const CellIndex &idx) const {
	    if(!isInitialized) return false;
	    if(!isInside(idx)) return false;
	    return (grid[idx.i][idx.j][idx.k] == SimpleOccMap::OCC);

	}
	
	virtual bool isOccupied(const Eigen::Vector3f &point) const;

	inline bool isFree(const CellIndex &idx) const {
	    if(!isInitialized) return false;
	    if(!isInside(idx)) return false;
	    return (grid[idx.i][idx.j][idx.k] == SimpleOccMap::FREE);

	}

	inline bool isUnknown(const CellIndex &idx) const {
	    if(!isInitialized) return false;
	    if(!isInside(idx)) return false;
	    return (grid[idx.i][idx.j][idx.k] == SimpleOccMap::UNKN);

	}

	///get indeces of all occupied cells
	void getOccupied(std::vector<CellIndex> &idx) const;
	void getFree(std::vector<CellIndex> &idx) const;
	void getUnknown(std::vector<CellIndex> &idx) const;

	///intersection methods
	void getIntersection(const SimpleOccMapIfce *other, std::vector<CellIndex> &idx_this) const;
	void getIntersectionWithPose(const SimpleOccMapIfce *other, Eigen::Affine3f &this_to_map, std::vector<CellIndex> &idx_this) const;

	///helpers
	inline bool isInside(const int &i, const int &j, const int &k) const {
	    if(!isInitialized) return false;
	    return ((i>=0) && (j>=0) && (k>=0) && (i<size_x) && (j<size_y) && (k<size_z)); 
	}
	
	inline bool isInside(const CellIndex &idx) const {
	    if(!isInitialized) return false;
	    return isInside(idx.i,idx.j,idx.k);
	}
	
	inline bool isInside(const Eigen::Vector3f &point) const {
	    if(!isInitialized) return false;
	    CellIndex idx;
	    return getIdxPoint(point,idx);

	}

	inline bool getIdxPoint(const Eigen::Vector3f &point, CellIndex &idx) const {
	    if(!isInitialized) {
		idx.i = -1; idx.j = -1; idx.k = -1;
		return false;
	    }
	    Eigen::Vector3f relToUpperLeft = (point-center + size_meters/2) / resolution;
	    idx.i = relToUpperLeft(0);
	    idx.j = relToUpperLeft(1);
	    idx.k = relToUpperLeft(2);

	    return isInside(idx); 
	}
	inline bool getCenterCell(const CellIndex &idx, Eigen::Vector3f &center_cell) const {
	    if(!isInside(idx)) return false;
	    Eigen::Vector3f rel;
	    rel<<idx.i,idx.j,idx.k;
	    center_cell = rel*resolution - size_meters/2 + center;
	    return true;
	}

	inline void resetMap() {
	    if(!isInitialized) return;
	    for (int i=0; i<size_x; ++i) {
		for (int j=0; j<size_y; ++j) {
		    for (int k=0; k<size_z; ++k) {
			grid[i][j][k] = SimpleOccMap::UNKN;
		    }
		}
	    }
	}

	inline void setAllFree() {
	    if(!isInitialized) return;
	    for (int i=0; i<size_x; ++i) {
		for (int j=0; j<size_y; ++j) {
		    for (int k=0; k<size_z; ++k) {
			grid[i][j][k] = SimpleOccMap::FREE;
		    }
		}
	    }
	}
	void toMessage(constraint_map::SimpleOccMapMsg &msg);
	void toPointCloud(pcl::PointCloud<pcl::PointXYZ> &pc);
	void fromMessage(const constraint_map::SimpleOccMapMsg &msg);
    public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


#endif
