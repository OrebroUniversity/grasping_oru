#include <constraint_map/region_extract_distance_field.h>

DistanceGrid::DistanceGrid(int sx, int sy, int sz) {
    size_x = sx;
    size_y = sy;
    size_z = sz;
    CellIndex id;

    distance_grid = new Triplet**[size_x];
    for(id.i = 0; id.i < size_x; ++id.i) {
	    distance_grid[id.i] = new Triplet*[size_y];
	    for(id.j = 0; id.j < size_y; ++id.j) {
		    distance_grid[id.i][id.j] = new Triplet[size_z];
	    }
    }
    isAlloc=true;
}

DistanceGrid::~DistanceGrid() {
    if(isAlloc) {
	//dealloc
	CellIndex id;
	for(id.i = 0; id.i < size_x; ++id.i) {
	    for(id.j = 0; id.j < size_y; ++id.j) {
		delete []  distance_grid[id.i][id.j];
	    }
	    delete [] distance_grid[id.i];
	}	    
	delete [] distance_grid;
	isAlloc=false;
    }
}

Triplet DistanceGrid::at(int i, int j, int k) {
    Triplet val(0,0,0);
    if(loopX) i = (i+size_x)%size_x;
    if(loopY) j = (j+size_y)%size_y;
    if(loopZ) k = (k+size_z)%size_z;

    /*if(i<0 && loopX) i = size_x - i;
    if(i>=size_x && loopX) i = i - size_x;
    if(j<0 && loopY) j = size_y - j;
    if(j>=size_y && loopY) j = j - size_y;
    if(k<0 && loopZ) k = size_z - k;
    if(k>=size_z && loopZ) k = k - size_z;*/

    if(i>=0 && i < size_x && j >=0 && j < size_y && k >=0 && k<size_z) {
	val = distance_grid[i][j][k];
    }
    return val; 
}

void DistanceGrid::computeDistanceGrid(SimpleOccMap *map) {

    if(map->size_x != size_x) return;
    if(map->size_y != size_y) return;
    if(map->size_z != size_z) return;

    CellIndex id;
    int borderI=1, borderJ=1, borderK=1;
    //pass through map: forward pass. initialize distances
    for(id.i = 0; id.i < map->size_x; ++id.i) {
	    distance_grid[id.i] = new Triplet*[map->size_y];
	    for(id.j = 0; id.j < map->size_y; ++id.j) {
		    distance_grid[id.i][id.j] = new Triplet[map->size_z];
		    for(id.k = 0; id.k < map->size_z; ++id.k) {
			    if(map->grid[id.i][id.j][id.k] > SimpleOccMap::FREE ) {
				    //std::cerr<<"OBST  "<<id.i<<" "<<id.j<<" "<<id.k<<std::endl; 
				    distance_grid[id.i][id.j][id.k][0] = 0;
				    distance_grid[id.i][id.j][id.k][1] = 0;
				    distance_grid[id.i][id.j][id.k][2] = 0;
			    } else {
				//find how far off is the border along X
				borderI=1, borderJ=1, borderK=1;
				if(loopX && id.i == 0) {
				    for(int idi = map->size_x-1; idi>=0; --idi) {
					if(map->grid[idi][id.j][id.k] > SimpleOccMap::FREE ) {
					    break;
					}
					borderI++;
				    }
				}
				if(loopY && id.j == 0) {
				    for(int idj = map->size_y-1; idj>=0; --idj) {
					if(map->grid[id.i][idj][id.k] > SimpleOccMap::FREE ) {
					    break;
					}
					borderJ++;
				    }
				}
				if(loopZ && id.k == 0) {
				    for(int idk = map->size_z-1; idk>=0; --idk) {
					if(map->grid[id.i][id.j][idk] > SimpleOccMap::FREE ) {
					    break;
					}
					borderK++;
				    }
				}
				//set it to 1+ prev value
				distance_grid[id.i][id.j][id.k][0] = minval(id.i > 0 ? distance_grid[id.i-1][id.j][id.k][0]+1 : borderI, map->size_x);
				distance_grid[id.i][id.j][id.k][1] = minval(id.j > 0 ? distance_grid[id.i][id.j-1][id.k][1]+1 : borderJ, map->size_y);
				distance_grid[id.i][id.j][id.k][2] = minval(id.k > 0 ? distance_grid[id.i][id.j][id.k-1][2]+1 : borderK, map->size_z);
			    }
		    }
	    }
    }	    
    //print outs
#if 0 
    for(id.i = 0; id.i < map->size_x; ++id.i) {
	    for(id.j = 0; id.j < map->size_y; ++id.j) {
		    for(id.k = 0; id.k < map->size_z; ++id.k) {
			    std::cout<<map->grid[id.i][id.j][id.k]<<"\t";
		    }
		    std::cout<<std::endl;
	    }
	    std::cout<<std::endl;
	    std::cout<<std::endl;
    }
    
    for(id.i = 0; id.i < map->size_x; ++id.i) {
	    for(id.j = 0; id.j < map->size_y; ++id.j) {
		    for(id.k = 0; id.k < map->size_z; ++id.k) {
			    //std::cout<<"("<<distance_grid[id.i][id.j][id.k][0]<<","<<distance_grid[id.i][id.j][id.k][1]<<","<<distance_grid[id.i][id.j][id.k][2]<<")\t";
			    std::cout<<distance_grid[id.i][id.j][id.k][1]<<","<<distance_grid[id.i][id.j][id.k][2]<<"\t";
		    }
		    std::cout<<std::endl;
	    }
	    std::cout<<std::endl;
	    std::cout<<std::endl;
    }
#endif
}

CellIdxCube DfunMaxEmptyCubeExtractor::getMaxCube(SimpleOccMap *map) {
    DistanceGrid distance_grid(map->size_x,map->size_y,map->size_z);
    distance_grid.loopX = loopX;
    distance_grid.loopY = loopY;
    distance_grid.loopZ = loopZ;
   
    distance_grid.computeDistanceGrid(map);
    CellIdxCube cube;
    CellIndex id;
    
    int maxvolume = 0;
    //pass through dist_grid
    for(id.i = map->size_x-1; id.i >=0; --id.i) {
	for(id.j = map->size_y-1; id.j >=0; --id.j) {
	    for(id.k = map->size_z-1; id.k >=0; --id.k) {
		Triplet thisone = distance_grid.at(id);
		if((thisone[0])*(thisone[1])*(thisone[2]) > maxvolume) {
		    //check what the maxvolume at this point would be
		    int best_c[2];
		    int best_v[3];	
		    Triplet min_ids[3];
		    for(int x=0; x<3; ++x) {
			min_ids[x] = thisone;
		    }

		    int mval;
		    bool goOn;

		    //search back to find the best feasible BL point
		    //////////////////////////////////////////////////along xy////////////////////////////////////////////////////
		    //reset values
		    best_c[0] = 1; best_c[1]=1;
		    best_v[0] = 0; best_v[1]=0; best_v[2]=0;
		    //along xy
		    for(int i=id.i; i>id.i-(min_ids[0][0]+1); --i) {
			mval = minval(min_ids[0][1],distance_grid.at(i,id.j,id.k)[1]);
			//is this the best so far?
			if( mval*(id.i-i+1) > best_c[1]*best_c[0]) {
			    best_c[0] = id.i-i+1;
			    best_c[1] = mval;
			}
			//can we potentially make a biger area than the best so far?
			if(mval*min_ids[0][0] > best_c[1]*best_c[0]) {
			    //yes we can, update the allowance
			    min_ids[0][1] = mval;
			} else {
			    //no, let's stick to what we found
			    min_ids[0][0] = best_c[0];
			    min_ids[0][1] = best_c[1];
			    break;
			}
		    }
		    //				std::cerr<<id.i<<" "<<id.j<<" "<<id.k<<": is :"<<thisone[0]<<" "<<thisone[1]<<" "<<thisone[2]<<std::endl;
		    //				std::cerr<<"bc (init) is "<<best_c[0]<<" "<<best_c[1]<<std::endl;
		    best_v[0] = best_c[0];
		    best_v[1] = best_c[1]; //for 0 depth cases
		    best_v[2] = 1;

		    //now let's sweep this cross section on the third dimension
		    goOn = true;
		    for(int k=id.k; k>id.k-(min_ids[0][2]+1)&&goOn; --k) {
			//find the largest cross section for this k index, within the constrained area
			best_c[0] = 1;
			best_c[1] = 1;
			for(int i=id.i; i>id.i-(min_ids[0][0]+1); --i) {
			    mval = minval(min_ids[0][1],distance_grid.at(i,id.j,k)[1]);
			    //is this the best so far?
			    if( mval*(id.i-i+1) > best_c[1]*best_c[0]) {
				best_c[0] = id.i-i+1;
				best_c[1] = mval;
			    }
			    //is this the best volume we found so far?
			    if(best_v[0]*best_v[1]*best_v[2] < best_c[0]*best_c[1]*(id.k-k+1)) {
				//yes -> update
				best_v[0] = best_c[0];
				best_v[1] = best_c[1];
				best_v[2] = id.k-k+1;
			    }

			    //can we potentially make a biger cross section area than the best so far?
			    if(mval*min_ids[0][0] > best_c[1]*best_c[0]) {
				//yes we can, continue looking
				min_ids[0][1] = mval;
			    } else {
				//no.
				//can we potentially find a larger volume?
				mval = minval(min_ids[0][2],distance_grid.at(i,id.j,k)[2]);
				if(best_c[0]*best_c[1]*mval > best_v[0]*best_v[1]*best_v[2]) {
				    //yes, update the boundaries 
				    min_ids[0][0] = best_c[0];
				    min_ids[0][1] = best_c[1];
				} else {
				    //no, break
				    goOn = false;
				}
				break;
			    }
			}
		    }
		    //				std::cerr<<" bv is "<<best_v[0]<<" "<<best_v[1]<<" "<<best_v[2]<<std::endl;
		    min_ids[0][0] = best_v[0];
		    min_ids[0][1] = best_v[1];
		    min_ids[0][2] = best_v[2];


		    //////////////////////////////////////////////////along xz////////////////////////////////////////////////////
		    //reset values
		    best_c[0] = 1; best_c[1]=1;
		    best_v[0] = 0; best_v[1]=0; best_v[2]=0;

		    //along xz
		    for(int i=id.i; i>id.i-(min_ids[1][0]+1); --i) {
			mval = minval(min_ids[1][2],distance_grid.at(i,id.j,id.k)[2]);
			//is this the best so far?
			if( mval*(id.i-i+1) > best_c[1]*best_c[0]) {
			    best_c[0] = id.i-i+1;
			    best_c[1] = mval;
			}
			//can we potentially make a biger area than the best so far?
			if(mval*min_ids[1][0] > best_c[1]*best_c[0]) {
			    //yes we can, update the allowance
			    min_ids[1][2] = mval;
			} else {
			    //no, let's stick to what we found
			    min_ids[1][0] = best_c[0];
			    min_ids[1][2] = best_c[1];
			    break;
			}
		    }
		    //				std::cerr<<id.i<<" "<<id.j<<" "<<id.k<<": is :"<<thisone[0]<<" "<<thisone[1]<<" "<<thisone[2]<<std::endl;
		    //				std::cerr<<"bc (init) is "<<best_c[0]<<" "<<best_c[1]<<std::endl;
		    best_v[0] = best_c[0];
		    best_v[1] = 1;
		    best_v[2] = best_c[1]; //for 0 depth cases

		    //now let's sweep this cross section on the third dimension
		    goOn = true;
		    for(int j=id.j; j>id.j-(min_ids[1][1]+1)&&goOn; --j) {
			//find the largest cross section for this k index, within the constrained area
			best_c[0] = 1;
			best_c[1] = 1;
			for(int i=id.i; i>id.i-(min_ids[1][0]+1); --i) {
			    mval = minval(min_ids[1][2],distance_grid.at(i,j,id.k)[2]);
			    //is this the best so far?
			    if( mval*(id.i-i+1) > best_c[1]*best_c[0]) {
				best_c[0] = id.i-i+1;
				best_c[1] = mval;
			    }
			    //is this the best volume we found so far?
			    if(best_v[0]*best_v[1]*best_v[2] < best_c[0]*best_c[1]*(id.j-j+1)) {
				//yes -> update
				best_v[0] = best_c[0];
				best_v[1] = id.j-j+1;
				best_v[2] = best_c[1];
			    }
			    //can we potentially make a biger area than the best so far?
			    if(mval*min_ids[1][0] > best_c[1]*best_c[0]) {
				//yes we can, update the allowance
				min_ids[1][2] = mval;
			    } else {
				//no.
				//					    std::cerr<<k<<" bc is "<<best_c[0]<<" "<<best_c[1]<<std::endl;
				//can we potentially find a larger volume?
				mval = minval(min_ids[1][1],distance_grid.at(i,j,id.k)[1]);
				if(best_c[0]*best_c[1]*mval > best_v[0]*best_v[1]*best_v[2]) {
				    //yes, update the boundaries 
				    min_ids[1][0] = best_c[0];
				    min_ids[1][2] = best_c[2];
				} else {
				    //no, break
				    goOn = false;
				}
				break;
			    }
			}
		    }
		    //				std::cerr<<" bv is "<<best_v[0]<<" "<<best_v[1]<<" "<<best_v[2]<<std::endl;
		    min_ids[1][0] = best_v[0];
		    min_ids[1][1] = best_v[1];
		    min_ids[1][2] = best_v[2];


		    //////////////////////////////////////////////////along yz////////////////////////////////////////////////////
		    //reset values
		    best_c[0] = 1; best_c[1]=1;
		    best_v[0] = 0; best_v[1]=0; best_v[2]=0;
		    for(int j=id.j; j>id.j-(min_ids[2][1]+1); --j) {
			mval = minval(min_ids[2][2],distance_grid.at(id.i,j,id.k)[2]);
			//is this the best so far?
			if( mval*(id.j-j+1) > best_c[1]*best_c[0]) {
			    best_c[0] = id.j-j+1;
			    best_c[1] = mval;
			}
			//can we potentially make a biger area than the best so far?
			if(mval*min_ids[2][1] > best_c[1]*best_c[0]) {
			    //yes we can, update the allowance
			    min_ids[2][2] = mval;
			} else {
			    //no, let's stick to what we found
			    min_ids[2][1] = best_c[0];
			    min_ids[2][2] = best_c[1];
			    break;
			}
		    }
		    //				std::cerr<<id.i<<" "<<id.j<<" "<<id.k<<": is :"<<thisone[0]<<" "<<thisone[1]<<" "<<thisone[2]<<std::endl;
		    //				std::cerr<<"bc (init) is "<<best_c[0]<<" "<<best_c[1]<<std::endl;
		    best_v[0] = 1;
		    best_v[1] = best_c[0];
		    best_v[2] = best_c[1]; //for 0 depth cases

		    //now let's sweep this cross section on the third dimension
		    goOn = true;
		    for(int i=id.i; i>id.i-(min_ids[2][0]+1)&&goOn; --i) {
			//find the largest cross section for this k index, within the constrained area
			best_c[0] = 1;
			best_c[1] = 1;
			for(int j=id.j; j>id.j-(min_ids[2][1]+1); --j) {
			    mval = minval(min_ids[2][2],distance_grid.at(i,j,id.k)[2]);
			    //is this the best so far?
			    if( mval*(id.j-j+1) > best_c[1]*best_c[0]) {
				best_c[0] = id.j-j+1;
				best_c[1] = mval;
			    }
			    //is this the best volume we found so far?
			    if(best_v[0]*best_v[1]*best_v[2] < best_c[0]*best_c[1]*(id.i-i+1)) {
				//yes -> update
				best_v[0] = id.i-i+1;
				best_v[1] = best_c[0];
				best_v[2] = best_c[1];
			    }
			    //can we potentially make a biger area than the best so far?
			    if(mval*min_ids[2][1] > best_c[1]*best_c[0]) {
				//yes we can, update the allowance
				min_ids[2][2] = mval;
			    } else {
				//no.
				//					    std::cerr<<k<<" bc is "<<best_c[0]<<" "<<best_c[1]<<std::endl;
				//can we potentially find a larger volume?
				mval = minval(min_ids[2][0],distance_grid.at(i,j,id.k)[0]);
				if(best_c[0]*best_c[1]*mval > best_v[0]*best_v[1]*best_v[2]) {
				    //yes, update the boundaries 
				    min_ids[2][1] = best_c[0];
				    min_ids[2][2] = best_c[2];
				} else {
				    //no, break
				    goOn = false;
				}
				break;
			    }
			}
		    }
		    //				std::cerr<<" bv is "<<best_v[0]<<" "<<best_v[1]<<" "<<best_v[2]<<std::endl;
		    min_ids[2][0] = best_v[0];
		    min_ids[2][1] = best_v[1];
		    min_ids[2][2] = best_v[2];


		    int maxvolu=-1;
		    int maxvolu_id=-1;
		    for(int x=0; x<3; ++x) {
			if(min_ids[x][0] < 0) continue;
			if(min_ids[x][1] < 0) continue;
			if(min_ids[x][2] < 0) continue;
			int v = (min_ids[x][0])*(min_ids[x][1])*(min_ids[x][2]);
			if(v >maxvolu) {
			    maxvolu = v;
			    maxvolu_id = x;
			}
		    }
		    if(maxvolu > maxvolume && maxvolu_id >=0) {
			maxvolume = maxvolu;
			cube.bl.i = id.i - (min_ids[maxvolu_id][0]-1);//
			cube.bl.j = id.j - (min_ids[maxvolu_id][1]-1);//
			cube.bl.k = id.k - (min_ids[maxvolu_id][2]-1);//
			cube.ur.i = id.i;
			cube.ur.j = id.j;
			cube.ur.k = id.k;
			//std::cerr<<"NEW "<<maxvolu_id<<" "<<maxvolume<<" "<<id.i<<" "<<id.j<<" "<<id.k<<": is :"<<min_ids[maxvolu_id][0]<<" "<<min_ids[maxvolu_id][1]<<" "<<min_ids[maxvolu_id][2]<<" was: "<<thisone[0]<<" "<<thisone[1]<<" "<<thisone[2]<<std::endl;
			//std::cerr<<"CUBE ("<<cube.bl.i<<","<<cube.bl.j<<","<<cube.bl.k<<") : ("<<cube.ur.i<<","<<cube.ur.j<<","<<cube.ur.k<<") volume "<<cube.volume()<<std::endl;
		    }

		}
	    }
	}
    }
    if(maxvolume == 0) {
	cube.bl.i = 0;
	cube.bl.j = 0;
	cube.bl.k = 0;
	cube.ur.i = 0;
	cube.ur.j = 0;
	cube.ur.k = 0;
    }


    return cube;
}

//This is an older implementation. It is not used anywhere and probably there is something horribly wrong with it.
// it is kept here just in case it's needed for some reason
#if 0
void DistanceGrid::computeDistanceGrid(SimpleOccMap *map) {

    if(map->size_x != size_x) return;
    if(map->size_y != size_y) return;
    if(map->size_z != size_z) return;

    CellIndex id;
    int borderI=1, borderJ=1, borderK=1;
    //pass through map: forward pass. initialize distances
    for(id.i = 0; id.i < map->size_x; ++id.i) {
	    distance_grid[id.i] = new Triplet*[map->size_y];
	    for(id.j = 0; id.j < map->size_y; ++id.j) {
		    distance_grid[id.i][id.j] = new Triplet[map->size_z];
		    for(id.k = 0; id.k < map->size_z; ++id.k) {
			    if(map->grid[id.i][id.j][id.k] > SimpleOccMap::FREE ) {
				    //std::cerr<<"OBST  "<<id.i<<" "<<id.j<<" "<<id.k<<std::endl; 
				    distance_grid[id.i][id.j][id.k][0] = 0;
				    distance_grid[id.i][id.j][id.k][1] = 0;
				    distance_grid[id.i][id.j][id.k][2] = 0;
			    } else {
				//find how far off is the border along X
				borderI=1, borderJ=1, borderK=1;
				if(loopX && id.i == 0) {
				    for(int idi = map->size_x-1; idi>=0; --idi) {
					if(map->grid[idi][id.j][id.k] > SimpleOccMap::FREE ) {
					    break;
					}
					borderI++;
				    }
				}
				if(loopY && id.j == 0) {
				    for(int idj = map->size_y-1; idj>=0; --idj) {
					if(map->grid[id.i][idj][id.k] > SimpleOccMap::FREE ) {
					    break;
					}
					borderJ++;
				    }
				}
				if(loopZ && id.k == 0) {
				    for(int idk = map->size_z-1; idk>=0; --idk) {
					if(map->grid[id.i][id.j][idk] > SimpleOccMap::FREE ) {
					    break;
					}
					borderK++;
				    }
				}
				//set it to 1+ prev value
				distance_grid[id.i][id.j][id.k][0] = id.i > 0 ? distance_grid[id.i-1][id.j][id.k][0]+1 : borderI;
				distance_grid[id.i][id.j][id.k][1] = id.j > 0 ? distance_grid[id.i][id.j-1][id.k][1]+1 : borderJ;
				distance_grid[id.i][id.j][id.k][2] = id.k > 0 ? distance_grid[id.i][id.j][id.k-1][2]+1 : borderK;
			    }
		    }
	    }
    }	    
    //reverse pass: set to min of previous value and current value
    for(id.i = map->size_x-1; id.i>=0; --id.i) {
	    for(id.j = map->size_y-1; id.j>=0; --id.j) {
		    for(id.k = map->size_z-1; id.k>=0; --id.k) {
			    if(map->grid[id.i][id.j][id.k] > SimpleOccMap::FREE ) {
				    //std::cerr<<"OBST2 "<<id.i<<" "<<id.j<<" "<<id.k<<std::endl; 
				    distance_grid[id.i][id.j][id.k][0] = 0;
				    distance_grid[id.i][id.j][id.k][1] = 0;
				    distance_grid[id.i][id.j][id.k][2] = 0;
			    } else {
				borderI=1, borderJ=1, borderK=1;
				if(loopX && id.i == map->size_x-1) {
				    for(int idi = 0; idi < map->size_x; ++idi) {
					if(map->grid[idi][id.j][id.k] > SimpleOccMap::FREE ) {
					    break;
					}
					borderI++;
				    }
				}
				if(loopY && id.j == map->size_y-1) {
				    for(int idj = 0; idj < map->size_y; ++idj) {
					if(map->grid[id.i][idj][id.k] > SimpleOccMap::FREE ) {
					    break;
					}
					borderJ++;
				    }
				}
				if(loopZ && id.k == map->size_z-1) {
				    for(int idk = 0; idk < map->size_z; ++idk) {
					if(map->grid[id.i][id.j][idk] > SimpleOccMap::FREE ) {
					    break;
					}
					borderK++;
				    }
				}
				//set it to 1+ prev value
				distance_grid[id.i][id.j][id.k][0] = minval(id.i < map->size_x-1 ? 
				    minval(distance_grid[id.i][id.j][id.k][0],distance_grid[id.i+1][id.j][id.k][0]+1) : minval(distance_grid[id.i][id.j][id.k][0],borderI), map->size_x/2+1);
				distance_grid[id.i][id.j][id.k][1] = minval(id.j < map->size_y-1 ?
				    minval(distance_grid[id.i][id.j][id.k][1],distance_grid[id.i][id.j+1][id.k][1]+1) : minval(distance_grid[id.i][id.j][id.k][1],borderJ), map->size_y/2+1);
				distance_grid[id.i][id.j][id.k][2] = minval(id.k < map->size_z-1 ?
				    minval(distance_grid[id.i][id.j][id.k][2],distance_grid[id.i][id.j][id.k+1][2]+1) : minval(distance_grid[id.i][id.j][id.k][2],borderK), map->size_z/2+1);
			    }
		    }
	    }
    }	   
    //print outs
#if 0 
    for(id.i = 0; id.i < map->size_x; ++id.i) {
	    for(id.j = 0; id.j < map->size_y; ++id.j) {
		    for(id.k = 0; id.k < map->size_z; ++id.k) {
			    std::cout<<map->grid[id.i][id.j][id.k]<<"\t";
		    }
		    std::cout<<std::endl;
	    }
	    std::cout<<std::endl;
	    std::cout<<std::endl;
    }
    
    for(id.i = 0; id.i < map->size_x; ++id.i) {
	    for(id.j = 0; id.j < map->size_y; ++id.j) {
		    for(id.k = 0; id.k < map->size_z; ++id.k) {
			    //std::cout<<"("<<distance_grid[id.i][id.j][id.k][0]<<","<<distance_grid[id.i][id.j][id.k][1]<<","<<distance_grid[id.i][id.j][id.k][2]<<")\t";
			    std::cout<<distance_grid[id.i][id.j][id.k][1]<<","<<distance_grid[id.i][id.j][id.k][2]<<"\t";
		    }
		    std::cout<<std::endl;
	    }
	    std::cout<<std::endl;
	    std::cout<<std::endl;
    }
#endif


}

CellIdxCube DfunMaxEmptyCubeExtractor::getMaxCube(SimpleOccMap *map) {
    DistanceGrid distance_grid(map->size_x,map->size_y,map->size_z);
    distance_grid.loopX = loopX;
    distance_grid.loopY = loopY;
    distance_grid.loopZ = loopZ;
   
    distance_grid.computeDistanceGrid(map);
    CellIdxCube cube;
    CellIndex id;
    
    /*Triplet t;
    CellIndex inde;
    inde.i=42; inde.j=49; inde.k=23;
    t = distance_grid.at(inde);
    std::cerr<<"2way Distance at culprit is"<<t[0]<<" "<<t[1]<<" "<<t[2]<<std::endl;
    inde.i=49; inde.j=48; inde.k=24;
    t = distance_grid.at(inde);
    std::cerr<<"2way Distance at chosen is"<<t[0]<<" "<<t[1]<<" "<<t[2]<<std::endl;*/
    
    int maxvolume = 0;
    //pass through dist_grid
    for(id.i = 0; id.i < map->size_x; ++id.i) {
	    for(id.j = 0; id.j < map->size_y; ++id.j) {
		    for(id.k = 0; id.k < map->size_z; ++id.k) {
			    Triplet thisone = distance_grid.at(id);
			    if((thisone[0])*(thisone[1])*(thisone[2]) > maxvolume) {
				//check what the maxvolume at this point would be
				Triplet min_ids[6];
				for(int x=0; x<6; ++x) {
				    min_ids[x] = thisone;
				}

				//along xy
				for(int i=id.i-(thisone[0]-1); i<id.i+(thisone[0]); ++i) {
					min_ids[0][1] = minval(min_ids[0][1],distance_grid.at(i,id.j,id.k)[1]);
				}
				for(int i=id.i-(thisone[0]-1); i<id.i+(thisone[0]); ++i) {
					for(int j=id.j-(min_ids[0][1]-1); j<id.j+(min_ids[0][1]); ++j) {
						min_ids[0][2] = minval(min_ids[0][2],distance_grid.at(i,j,id.k)[2]);
					}
				}
				//along xz
				for(int i=id.i-(thisone[0]-1); i<id.i+(thisone[0]); ++i) {
					min_ids[1][2] = minval(min_ids[1][2],distance_grid.at(i,id.j,id.k)[2]);
				}
				for(int i=id.i-(thisone[0]-1); i<id.i+(thisone[0]); ++i) {
					for(int k=id.k-(min_ids[1][2]-1); k<id.k+(min_ids[1][2]); ++k) {
						min_ids[1][1] = minval(min_ids[1][1],distance_grid.at(i,id.j,k)[1]);
					}
				}

				//along yx
				for(int j=id.j-(thisone[1]-1); j<id.j+(thisone[1]); ++j) {
					min_ids[2][0] = minval(min_ids[2][0],distance_grid.at(id.i,j,id.k)[0]);
				}
				for(int j=id.j-(thisone[1]-1); j<id.j+(thisone[1]); ++j) {
					for(int i=id.i-(min_ids[2][0]-1); i<id.i+(min_ids[2][0]); ++i) {
						min_ids[2][2] = minval(min_ids[2][2],distance_grid.at(i,j,id.k)[2]);
					}
				}
				//along yz
				for(int j=id.j-(thisone[1]-1); j<id.j+(thisone[1]); ++j) {
					min_ids[3][2] = minval(min_ids[3][2],distance_grid.at(id.i,j,id.k)[2]);
				}
				for(int j=id.j-(thisone[1]-1); j<id.j+(thisone[1]); ++j) {
					for(int k=id.k-(min_ids[3][2]-1); k<id.k+(min_ids[3][2]); ++k) {
						min_ids[3][0] = minval(min_ids[3][0],distance_grid.at(id.i,j,k)[0]);
					}
				}

				//along zx
				for(int k=id.k-(thisone[2]-1); k<id.k+(thisone[2]); ++k) {
					min_ids[4][0] = minval(min_ids[4][0],distance_grid.at(id.i,id.j,k)[0]);
				}
				for(int k=id.k-(thisone[2]-1); k<id.k+(thisone[2]); ++k) {
					for(int i=id.i-(min_ids[4][0]-1); i<id.i+(min_ids[4][0]); ++i) {
						min_ids[4][1] = minval(min_ids[4][1],distance_grid.at(i,id.j,k)[1]);
					}
				}
				//along zy
				for(int k=id.k-(thisone[2]-1); k<id.k+(thisone[2]); ++k) {
					min_ids[5][1] = minval(min_ids[5][1],distance_grid.at(id.i,id.j,k)[1]);
				}
				for(int k=id.k-(thisone[2]-1); k<id.k+(thisone[2]); ++k) {
					for(int j=id.j-(min_ids[5][1]-1); j<id.j+(min_ids[5][1]); ++j) {
						min_ids[5][0] = minval(min_ids[5][0],distance_grid.at(id.i,j,k)[0]);
					}
				}
					
				int maxvolu=-1;
				int maxvolu_id=-1;
				for(int x=0; x<6; ++x) {
					if(min_ids[x][0] < 0) continue;
					if(min_ids[x][1] < 0) continue;
					if(min_ids[x][2] < 0) continue;
					int v = (min_ids[x][0])*(min_ids[x][1])*(min_ids[x][2]);
					if(v >maxvolu) {
						maxvolu = v;
						maxvolu_id = x;
					}
				}
				if(maxvolu > maxvolume && maxvolu_id >=0) {
					maxvolume = maxvolu;
					cube.bl.i = id.i - (min_ids[maxvolu_id][0]-1);//);
					cube.bl.j = id.j - (min_ids[maxvolu_id][1]-1);//);
					cube.bl.k = id.k - (min_ids[maxvolu_id][2]-1);//);
					cube.ur.i = id.i + (min_ids[maxvolu_id][0]-1);//);
					cube.ur.j = id.j + (min_ids[maxvolu_id][1]-1);//);
					cube.ur.k = id.k + (min_ids[maxvolu_id][2]-1);//);
					//std::cerr<<"NEW "<<maxvolu_id<<" "<<maxvolume<<" "<<id.i<<" "<<id.j<<" "<<id.k<<": is :"<<min_ids[maxvolu_id][0]<<" "<<min_ids[maxvolu_id][1]<<" "<<min_ids[maxvolu_id][2]<<" was: "<<thisone[0]<<" "<<thisone[1]<<" "<<thisone[2]<<std::endl;
					//std::cerr<<"CUBE ("<<cube.bl.i<<","<<cube.bl.j<<","<<cube.bl.k<<") : ("<<cube.ur.i<<","<<cube.ur.j<<","<<cube.ur.k<<") volume "<<cube.volume()<<std::endl;
				}

			    }
		    }
	    }
    }
    if(maxvolume == 0) {
	cube.bl.i = 0;
	cube.bl.j = 0;
	cube.bl.k = 0;
	cube.ur.i = 0;
	cube.ur.j = 0;
	cube.ur.k = 0;
    }

    return cube;
}

CellIdxCube DfunMaxEmptyCubeExtractor::bruteForceMaxCube(SimpleOccMap *map) {


}

#endif
