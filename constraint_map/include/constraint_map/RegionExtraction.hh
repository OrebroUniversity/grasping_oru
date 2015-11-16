#ifndef _REGION_EXTRACTION_HH
#define _REGION_EXTRACTION_HH
/**The classes in this header implement the method for maximal empty hyper rectangle extraction by Datta, with some simplifications */
#include <constraint_map/SimpleOccMap.hh>
#include <boost/intrusive/rbtree.hpp>

class SegmentNode : public boost::intrusive::set_base_hook< >{
    public:
	int x, y, xdir, ydir;
	SegmentNode(int x_, int y_, int xdir_, int ydir_):x(x_),y(y_),xdir(xdir_),ydir(ydir_)
	{}
	virtual ~SegmentNode() { }
	friend bool operator< (const SegmentNode &a, const SegmentNode &b)
	{  return a.xdir*a.x < b.xdir*b.x;  }
	friend bool operator> (const SegmentNode &a, const SegmentNode &b)
	{  return a.xdir*a.x > b.xdir*b.x;  }
	friend bool operator== (const SegmentNode &a, const SegmentNode &b)
	{  return a.xdir*a.x == b.xdir*b.x;  }
};

inline bool SegmentYCmp(const SegmentNode &a, const SegmentNode &b)
	{  return a.ydir*a.y >= b.ydir*b.y;  }

struct delete_disposer
{
       void operator()(SegmentNode *delete_this)
	      {  delete delete_this;  }
};
//Cloner object function
struct new_cloner
{
       SegmentNode *operator()(const SegmentNode &clone_this)
	      {  return new SegmentNode(clone_this.x, clone_this.y, clone_this.xdir, clone_this.ydir);  }
};


class SegmentTree {
    private:
    public:
	int minx, maxx, miny, maxy;
	int xdir, ydir;
	boost::intrusive::rbtree<SegmentNode> tree;

	SegmentTree (int minx_, int maxx_, int miny_, int maxy_, int xdir_, int ydir_);
	SegmentTree (const SegmentTree &other) {
	    minx = other.minx;
	    maxx = other.maxx;
	    miny = other.miny;
	    maxy = other.maxy;
	    xdir = other.xdir;
	    ydir = other.ydir;
	    tree.clone_from(other.tree, new_cloner(), delete_disposer());
	}

	virtual ~SegmentTree() {
	    tree.clear_and_dispose(delete_disposer());
	}
	bool isActive (int &x, int &y);
        void insertPoint(int &x, int &y);
	void printTree() {
	    boost::intrusive::rbtree<SegmentNode>::iterator it = tree.begin();
	    while (it!= tree.end()) {
		std::cerr<<"("<<it->x<<","<<it->y<<") ";
		it++;
	    }
	    std::cerr<<std::endl;
	}	    
};


class ActiveSet {
    public:
	//coordinates of the point
	int x, y, z;
	int minx, maxx, miny, maxy;
	SegmentTree *bl, *br, *ul, *ur; //bottom-left, bottom-right, upper-left and upper-right stairs
	//////
	ActiveSet(int x_, int y_, int z_, int minx_, int maxx_, int miny_, int maxy_);
	virtual ~ActiveSet() {
	    delete bl;
	    delete br;
	    delete ul;
	    delete ur;
	}
	bool isActive (int &x, int &y);
        void insertPoint(int &x, int &y);
	//computes the max area rectangle in the active set and stores bottom left/ upper right corners. returns area.
	int maxRectangle(int &xbl, int &ybl, int &xur, int &yur); 
};

class CellIdxCube {
    public:
	CellIndex bl, ur;
	int volume() const{
	   return (((ur.i-bl.i)+1)*((ur.j-bl.j)+1)*((ur.k-bl.k)+1)); 
	}
};
inline bool cubecmpr(const CellIdxCube &a, const CellIdxCube &b) {
    return a.volume() > b.volume();
}

class MaxEmptyCubeExtractor {

    public:
	std::vector<CellIdxCube> empty_cubes;
	std::vector<ActiveSet*> visited;

	CellIdxCube getMaxCube(SimpleOccMap *map);

};

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
    void computeDistanceGrid(SimpleOccMap *map);
    void computeDistanceGrid2(SimpleOccMap *map);
};

class DfunMaxEmptyCubeExtractor {
    public:
	std::vector<CellIdxCube> empty_cubes;
	bool loopX, loopY, loopZ;
	DfunMaxEmptyCubeExtractor():loopX(false),loopY(false),loopZ(false) {};
	CellIdxCube getMaxCube(SimpleOccMap *map);
	CellIdxCube getMaxCube2(SimpleOccMap *map);
	CellIdxCube bruteForceMaxCube(SimpleOccMap *map);
};


#endif
