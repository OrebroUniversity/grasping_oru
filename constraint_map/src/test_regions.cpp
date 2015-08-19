#include <constraint_map/RegionExtraction.hh>
#include <iostream>
#include <constraint_map/ConstraintMap.hh>
using namespace std;

double getDoubleTime()
{
    struct timeval time;
    gettimeofday(&time,NULL);
    return time.tv_sec + time.tv_usec * 1e-6;
}
int main() {
#if 0
#endif
    int map_size = 50;
    float resolution = 1;
    SimpleOccMap sm(0,0,0,resolution,map_size,map_size,map_size);
    sm.setAllFree();

    for(int j=0; j<10; ++j) {

	for(int i=0; i<500; ++i) {
	    int x = (float)rand()* map_size/RAND_MAX; 
	    int y = (float)rand()* map_size/RAND_MAX; 
	    int z = (float)rand()* map_size/RAND_MAX;
	    sm.setOccupied(x,y,z);	
	}
	double t1 = getDoubleTime();
	//MaxEmptyCubeExtractor extractor;
	DfunMaxEmptyCubeExtractor extractor;
	extractor.loopX = true;
	extractor.loopY = true;
	extractor.loopZ = true;
	CellIdxCube cube;
	cube = extractor.getMaxCube(&sm);
	double t2 = getDoubleTime();
	std::cout<<"extract took :"<<t2-t1<<" sec\n";
	std::cout<<"MAX cube at ("<<cube.bl.i<<","<<cube.bl.j<<","<<cube.bl.k<<") : ("<<cube.ur.i<<","<<cube.ur.j<<","<<cube.ur.k<<") volume "<<cube.volume()<<endl;
	

	bool isokay = true;
	CellIndex id, id2;
	for(id.i = cube.bl.i; id.i<=cube.ur.i; ++id.i) {
		for(id.j = cube.bl.j; id.j<=cube.ur.j; ++id.j) {
			for(id.k = cube.bl.k; id.k<=cube.ur.k; ++id.k) {
				id2.i = (id.i + map_size)%map_size;
				id2.j = (id.j + map_size)%map_size;
				id2.k = (id.k + map_size)%map_size;
				if(sm.isOccupied(id2)) {
					std::cout<<id2.i<<","<<id2.j<<","<<id2.k<<" really? "<<sm.isOccupied(id2)<<std::endl;
					isokay=false;
				}
			}
		}
	}
	if(isokay) std::cerr<<"OKAY\n";
/* 
        ConstraintMap object_map (0,0,0,resolution,map_size,map_size,map_size);
	Eigen::Affine3f pose;
	pose.setIdentity();
	Eigen::Vector3f bl,ur;
	if(!object_map.getCenterCell(cube.bl,bl)) {
	}
	if(!object_map.getCenterCell(cube.ur,ur)) {
	}
	std::cout<<"ur "<<ur.transpose()<<std::endl;
	pose.translation() = bl ;//<<cube.bl.i/resolution,cube.bl.j/resolution,cube.bl.k/resolution;
	Eigen::Vector3f box_size = ur-bl; //((cube.ur.i-cube.bl.i)/resolution,(cube.ur.j-cube.bl.j)/resolution,(cube.ur.k-cube.bl.k)/resolution);
	std::cerr<<"box size "<<box_size.transpose()<<std::endl;
	std::cerr<<"box pos "<<pose.translation().transpose()<<std::endl;
	object_map.drawBox(pose,box_size);
	object_map.updateMap();
	std::vector<CellIndex> ids;
	sm.getIntersection(&object_map, ids);
	if(ids.size() == 0) {
	    std::cout << "OK!\n";
	} else {
	    std::cout<< "Problematic indexes at: \n";
	    for(int q=0; q<ids.size(); ++q) {
		std::cout<<ids[q].i<<","<<ids[q].j<<","<<ids[q].k<<" really? "<<sm.isOccupied(ids[q])<<std::endl;
	    }
	}
*/
    }
#if 0
//test cases for active set
    ActiveSet as(8,8,0,0,16,0,16);
    //static const int arr_x[] = {2,2,7,9,10,10,11,11,12,13,13,14,14,5};
    //static const int arr_x[] = {2,3,4,8,8,12,13,14};
    static const int arr_x[] = {1,2,8,14,15};
    std::vector<int> x(arr_x, arr_x + sizeof(arr_x) / sizeof(arr_x[0]) );
    //static const int arr_y[] = {10,14,12,14,1,13,4,11,14,2,9,6,8,5};
    //static const int arr_y[] = {8,11,2,1,13,12,3,8};
    static const int arr_y[] = {3,15,1,14,7};
    std::vector<int> y(arr_y, arr_y + sizeof(arr_y) / sizeof(arr_y[0]) );
    std::cout<<"sane? "<<(x.size()==y.size())<<"\n";
    //for(int i=0; i<x.size() && i<y.size(); ++i) {
//	as.insertPoint(x[i],y[i]);
//    }
    cout<<"AS is: \n BL:\n";
    as.bl->printTree();
    std::cout<<"BR: \n";
    as.br->printTree();
    std::cout<<"UL: \n";
    as.ul->printTree();
    std::cout<<"UR: \n";
    as.ur->printTree();

    int xbl, ybl, xur, yur, area;
    area = as.maxRectangle(xbl,ybl,xur,yur);
    std::cout <<"Got max rectangle area "<<area<<" at ("<<xbl<<","<<ybl<<"):("<<xur<<","<<yur<<")\n";
#endif
//test cases for segment tree
#if 0	
    SegmentTree tr (0,12,0,6,-1,1);
    
    int x,y;
    x=2; y=4;
    tr.insertPoint(x,y);
    tr.printTree();
    
    x=4; y=3;
    tr.insertPoint(x,y);
    tr.printTree();
    
    x=6; y=2;
    tr.insertPoint(x,y);
    tr.printTree();
    
    x=7; y=4;
    tr.insertPoint(x,y);
    tr.printTree();
    
    x=7; y=5;
    tr.insertPoint(x,y);
    tr.printTree();
    
    x=8; y=1;
    tr.insertPoint(x,y);
    tr.printTree();
    
    x=9; y=5;
    tr.insertPoint(x,y);
    tr.printTree();
    
    x=2; y=4;
    tr.insertPoint(x,y);
    tr.printTree();
#endif

    return 0;
}
