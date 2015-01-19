#include <simple_occ_map/SimpleOccMap.hh>

SimpleOccMap::SimpleOccMap(float _cen_x, float _cen_y, float _cen_z, 
	float _resolution, int _size_x, int _size_y, int _size_z) {

    center(0) = _cen_x;
    center(1) = _cen_y;
    center(2) = _cen_z;
    resolution = _resolution;
    size_x = _size_x;
    size_y = _size_y;
    size_z = _size_z;
    size_meters << size_x,size_y,size_z;
    size_meters = size_meters * resolution;

    initialize();
}

void SimpleOccMap::initialize() {
    if(resolution < 0) return;
    grid = new int** [size_x];
    for (int i=0; i<size_x; ++i) {
	grid[i] = new int* [size_y];
	for (int j=0; j<size_y; ++j) {
	    grid[i][j] = new int[size_z];
	    for (int k=0; k<size_z; ++k) {
		grid[i][j][k] = SimpleOccMap::UNKN;
	    }
	}
    }
    isInitialized = true;
}
	
SimpleOccMap::SimpleOccMap(const SimpleOccMap& other) {
    if(other.isInitialized) {
	center = other.center;
	resolution = other.resolution;
	size_x = other.size_x;
	size_y = other.size_y;
	size_z = other.size_z;
	size_meters << size_x,size_y,size_z;
	size_meters = size_meters * resolution;
	initialize();
	for (int i=0; i<size_x; ++i) {
	    for (int j=0; j<size_y; ++j) {
		for (int k=0; k<size_z; ++k) {
		    grid[i][j][k] = other.grid[i][j][k];
		}
	    }
	}
    }
}
	
SimpleOccMap::~SimpleOccMap() {
    if(isInitialized && grid != NULL) {
	for (int i=0; i<size_x; ++i) {
	    for (int j=0; j<size_y; ++j) {
		delete[] grid[i][j];
	    }
	    delete[] grid[i];
	}
	delete[] grid;
    }
}

void SimpleOccMap::toMessage(simple_occ_map::SimpleOccMapMsg &msg) {
    if(!isInitialized) return;
    msg.header.frame_id = "my_frame";
    msg.cell_size = resolution;
    msg.x_cen = center(0);
    msg.y_cen = center(1);
    msg.z_cen = center(2);
    msg.x_size = size_x;
    msg.y_size = size_y;
    msg.z_size = size_z;
    for (int i=0; i<size_x; ++i) {
	for (int j=0; j<size_y; ++j) {
	    for (int k=0; k<size_z; ++k) {
		msg.data.push_back(grid[i][j][k]);
	    }
	}
    }

}

void SimpleOccMap::fromMessage(const simple_occ_map::SimpleOccMapMsg &msg) {
    
    center(0) = msg.x_cen;
    center(1) = msg.y_cen;
    center(2) = msg.z_cen;
    resolution = msg.cell_size; 
    size_x = msg.x_size;
    size_y = msg.y_size;
    size_z = msg.z_size;
    size_meters << size_x,size_y,size_z;
    size_meters = size_meters * resolution;

    initialize();
    int ctr = 0;
    for (int i=0; i<size_x; ++i) {
	for (int j=0; j<size_y; ++j) {
	    for (int k=0; k<size_z; ++k) {
		if(ctr >= msg.data.size()) return;
		grid[i][j][k] = msg.data[ctr];
		ctr++;
	    }
	}
    }

}
	
void SimpleOccMap::getOccupied(std::vector<CellIndex> &idx) const {
    CellIndex id;
    for (id.i=0; id.i<size_x; ++id.i) {
	for (id.j=0; id.j<size_y; ++id.j) {
	    for (id.k=0; id.k<size_z; ++id.k) {
		if(grid[id.i][id.j][id.k] == SimpleOccMap::OCC) idx.push_back(id);
	    }
	}
    }
}

void SimpleOccMap::getFree(std::vector<CellIndex> &idx) const {
    CellIndex id;
    for (id.i=0; id.i<size_x; ++id.i) {
	for (id.j=0; id.j<size_y; ++id.j) {
	    for (id.k=0; id.k<size_z; ++id.k) {
		if(grid[id.i][id.j][id.k] == SimpleOccMap::FREE) idx.push_back(id);
	    }
	}
    }
}

void SimpleOccMap::getUnknown(std::vector<CellIndex> &idx) const {
    CellIndex id;
    for (id.i=0; id.i<size_x; ++id.i) {
	for (id.j=0; id.j<size_y; ++id.j) {
	    for (id.k=0; id.k<size_z; ++id.k) {
		if(grid[id.i][id.j][id.k] == SimpleOccMap::UNKN) idx.push_back(id);
	    }
	}
    }

}

void SimpleOccMap::getIntersection(const SimpleOccMap *other, std::vector<CellIndex> &idx_this) const {

    //go through this map, compute corresponding index for other map and check if both occupied
    CellIndex id;
    for (id.i=0; id.i<size_x; ++id.i) {
	for (id.j=0; id.j<size_y; ++id.j) {
	    for (id.k=0; id.k<size_z; ++id.k) {
		if(grid[id.i][id.j][id.k] == SimpleOccMap::OCC) {
		    Eigen::Vector3f this_point;
		    CellIndex other_idx;
		    if(!this->getCenterCell(id,this_point)) continue;
		    if(other->getIdxPoint(this_point,other_idx)) {
			if(other->isOccupied(other_idx)) idx_this.push_back(id);
		    }
		}	   
	    }
	}
    }
    
}

void SimpleOccMap::getIntersectionWithPose(const SimpleOccMap *other, Eigen::Affine3f &this_to_map, std::vector<CellIndex> &idx_this) const {

    CellIndex id;
    for (id.i=0; id.i<size_x; ++id.i) {
	for (id.j=0; id.j<size_y; ++id.j) {
	    for (id.k=0; id.k<size_z; ++id.k) {
		if(grid[id.i][id.j][id.k] == SimpleOccMap::OCC) {
		    Eigen::Vector3f this_point;
		    CellIndex other_idx;
		    if(!this->getCenterCell(id,this_point)) continue;
		    this_point = this_to_map*this_point;
		    if(other->getIdxPoint(this_point,other_idx)) {
			if(other->isOccupied(other_idx)) idx_this.push_back(id);
		    }
		}	   
	    }
	}
    }
}
