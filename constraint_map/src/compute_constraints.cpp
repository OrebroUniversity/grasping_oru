#include <constraint_map/ConstraintMap.hh>

int main(int argc, char **argv) {

    if(argc !=4 ){
	std::cerr<<"usage: "<<argv[0]<<" filename resolution map_size_meters\n";
	return -1;
    }
    char *fname = argv[1];
    float resolution = atof(argv[2]);
    float map_size = atof(argv[3]);
    ConstraintMap *map;
    map = new ConstraintMap(0,0,0,resolution,map_size/resolution,map_size/resolution,map_size/resolution, false);
    map->sampleGripperGrid(10, 100, 7, 0.0, 0.1, 0.08, 0.14);
    //map->sampleGripperGrid(15, 100, 7, 0.0, 0.20, 0.0, 0.2);
    //map->sampleGripperGridSphere(15, 100, 7, 0.0, 0.2);
    map->updateMapAndGripperLookup(); 
    map->saveGripperConstraints(fname);
   
    delete map; 
    

    return 0;
}
