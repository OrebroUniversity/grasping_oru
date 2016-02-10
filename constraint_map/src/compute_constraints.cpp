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
    map = new ConstraintMap(0,0,0,resolution,map_size/resolution,map_size/resolution,map_size/resolution);
    map->sampleGripperGrid(15, 100, 7, 0.0, 0.20, 0.1, 0.3);
    //map->sampleGripperGridSphere(15, 100, 7, 0.1, 0.3);
    //map->sampleGripperGrid(10, 100, 20, 0.0, 0.4, 0.15, 0.5);
    //map->sampleGripperGrid(3, 10, 20, 0.1, 0.5, 0.2, 0.8);
    map->updateMapAndGripperLookup(); 
    map->saveGripperConstraints(fname);
   
    delete map; 
    

    return 0;
}
