#include <constraint_map/ConstraintMap.hh>
#include <cstdio>
void ConstraintMap::updateMap() {
#define n_threads 6
    double t1 = getDoubleTime();
    if(boxes.size() != 0 || cylinders.size() != 0) {
#pragma omp parallel num_threads(n_threads)
	{
	    #pragma omp for
	    for (unsigned int index=0; index<size_x; ++index) {
		CellIndex id;
		id.i = (int)index;
		for (id.j=0; id.j<size_y; ++id.j) {
		    for (id.k=0; id.k<size_z; ++id.k) {
			//if already occupied don't even check
			if(grid[id.i][id.j][id.k] == SimpleOccMap::OCC) continue;
			Eigen::Vector3f x;
			this->getCenterCell(id, x);
			for(int i=0; i<boxes.size(); ++i) {
			    if(boxes[i](x)) {
				grid[id.i][id.j][id.k] = SimpleOccMap::OCC;
				break;
			    }
			}	
			if(grid[id.i][id.j][id.k] == SimpleOccMap::OCC) continue;
			for(int i=0; i<cylinders.size(); ++i) {
			    if(cylinders[i](x)) {
				grid[id.i][id.j][id.k] = SimpleOccMap::OCC;
				break;
			    }
			}	
		    }
		}
	    }
	}
    }
    double t2 = getDoubleTime();

    std::cerr<<"Map update took "<<t2-t1<<" seconds for "<<boxes.size()<<" constraints\n";

}

void ConstraintMap::updateMapAndGripperLookup() {
#define n_threads 6
    double t1 = getDoubleTime();
#pragma omp parallel num_threads(n_threads)
    {
#pragma omp for
	for (unsigned int index=0; index<size_x; ++index) {
	    CellIndex id;
	    id.i = index;
	    for (id.j=0; id.j<size_y; ++id.j) {
		for (id.k=0; id.k<size_z; ++id.k) {
		    Eigen::Vector3f x;
		    this->getCenterCell(id, x);
		    for(int i=0; i<valid_configs.size(); ++i) {
			if(valid_configs[i]->palm(x) || valid_configs[i]->fingerSweep(x)) {
			    grid[id.i][id.j][id.k] = SimpleOccMap::OCC;
			    config_grid[id.i][id.j][id.k].configs.push_back(&valid_configs[i]);
			    config_grid[id.i][id.j][id.k].config_ids.push_back(i);
			}
		    }	
		}
	    }
	}
    }
    double t2 = getDoubleTime();

    std::cerr<<"Gripper map update took "<<t2-t1<<" seconds for "<<valid_configs.size()<<" constraints\n";

}
	
void ConstraintMap::drawValidConfigs() {
   
    if(isPJGripper) return; 
    double t1 = getDoubleTime();
    std::vector<GripperConfigurationSimple> simple_configs;
    for(int i=0; i<valid_configs.size(); ++i) {
	if(valid_configs[i] == NULL) continue;
	if(!valid_configs[i]->isValid) continue;
	GripperConfigurationSimple sc (valid_configs[i]->pose, valid_configs[i]->min_oa, model);
	sc.calculateConstraints();
	simple_configs.push_back(sc);
	GripperConfigurationSimple scm (valid_configs[i]->pose, valid_configs[i]->max_oa, model);
	scm.calculateConstraints();
	simple_configs.push_back(scm);
    }

#define n_threads 6
#pragma omp parallel num_threads(n_threads)
    {
#pragma omp for
	for (unsigned int index=0; index<size_x; ++index) {
	    CellIndex id;
	    id.i = index;
	    for (id.j=0; id.j<size_y; ++id.j) {
		for (id.k=0; id.k<size_z; ++id.k) {
		    Eigen::Vector3f x;
		    this->getCenterCell(id, x);
		    for(int i=0; i<simple_configs.size(); ++i) {
			if(simple_configs[i].palm(x) || simple_configs[i].leftFinger(x) || simple_configs[i].rightFinger(x)) {
			    grid[id.i][id.j][id.k] = SimpleOccMap::OCC;
			}
		    }	
		}
	    }
	}
    }
    double t2 = getDoubleTime();

    std::cerr<<"Drawing valid configs took "<<t2-t1<<" seconds for "<<simple_configs.size()<<" constraints\n";

}
	
void ConstraintMap::getConfigsForDisplay(pcl::PointCloud<pcl::PointXYZRGB> &configs_pc) {
    
    CellIndex id;
    Eigen::Affine3f pose;
    pcl::PointXYZRGB pc, valid, select, invalid;
    // pack r/g/b into rgb
    uint8_t r = 255, g = 0, b = 0;    // Red color
    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
    invalid.rgb = *reinterpret_cast<float*>(&rgb);
    r = 0, g = 250, b = 0;    // Green color
    rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
    select.rgb = *reinterpret_cast<float*>(&rgb);
    r = 100, g = 50, b = 20;    // Orange color
    rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
    valid.rgb = *reinterpret_cast<float*>(&rgb);

    int lowY, highY;
    lowY = (n_o+cube.bl.j)%n_o;
    highY = (n_o+cube.ur.j)%n_o;
    int lowX, highX;
    lowX = (n_v+cube.bl.i)%n_v;
    highX = (n_v+cube.ur.i)%n_v;
    //printf("bounds were: %d,%d adjusted are: %d,%d no=%d",cube.bl.j,cube.ur.j,lowY,highY,n_o);
    //the non-selected valid configs in green...
    for (id.i=0; id.i<n_v; ++id.i) {
	for (id.j=0; id.j<n_o; ++id.j) {
	    for (id.k=0; id.k<n_d; ++id.k) {
		getPoseForConfig(id,pose);
		pc.x = pose.translation()(0);
		pc.y = pose.translation()(1);
		pc.z = pose.translation()(2);
		
		int config_index =  id.k + n_d*id.j + n_d*n_o*id.i;
		if(valid_configs[config_index] == NULL) continue;
		if(valid_configs[config_index]->isValid) {
		    //valid
		    if( ( id.k >= cube.bl.k && id.k<=cube.ur.k) && //non-looped degreed
			( isSphereGrid ? (lowX<=highX ? (id.i >= lowX && id.i<= highX) :  (id.i >= lowX || id.i<= highX )) : (id.i >= cube.bl.i && id.i<=cube.ur.i) ) && 
			(  lowY<=highY ? (id.j >= lowY && id.j<= highY) :  (id.j >= lowY || id.j<= highY ) ) ) {
			//selected
			pc.rgb = select.rgb;
		    } else {
			//not selected
			pc.rgb = valid.rgb;
		    }	
		    configs_pc.points.push_back(pc);
		} else {
		    //invalid
		    if( ( id.k >= cube.bl.k && id.k<=cube.ur.k) && //non-looped degreed
			( isSphereGrid ? (lowX<=highX ? (id.i >= lowX && id.i<= highX) :  (id.i >= lowX || id.i<= highX )) : (id.i >= cube.bl.i && id.i<=cube.ur.i) ) && 
			(  lowY<=highY ? (id.j >= lowY && id.j<= highY) :  (id.j >= lowY || id.j<= highY ) ) ) {
			pc.rgb = invalid.rgb;
			configs_pc.points.push_back(pc);
		      }
		}
	    }
	}
    }
    configs_pc.is_dense=false;
    configs_pc.width = configs_pc.points.size();
    configs_pc.height = 1;
}
	
void ConstraintMap::generateOpeningAngleDump(std::string &fname) {
    
    FILE *fout = fopen(fname.c_str(), "w+");
    if(fout == NULL) {
	std::cerr<<"couldn't open file for writing at "<<fname<<std::endl;
	return;
    }

    CellIndex id;
    
    int lowY, highY;
    lowY = (n_o+cube.bl.j)%n_o;
    highY = (n_o+cube.ur.j)%n_o;
    int lowX, highX;
    lowX = (n_v+cube.bl.i)%n_v;
    highX = (n_v+cube.ur.i)%n_v;
    
    for (id.k=0; id.k<n_d; ++id.k) {
	fprintf(fout,"configs(:,:,%d) = [", id.k+1);
	for (id.j=0; id.j<n_o; ++id.j) {
	    for (id.i=0; id.i<n_v; ++id.i) {
		
		if(id.i>0) fprintf(fout,", ");

		int config_index =  id.k + n_d*id.j + n_d*n_o*id.i;
		if(valid_configs[config_index] == NULL) {
		    fprintf(fout,"-1");
		    continue;
		}
		if(valid_configs[config_index]->isValid) {
		    fprintf(fout,"%f",valid_configs[config_index]->min_oa);
		} else {
		    fprintf(fout,"0");
		}
	    }
	    fprintf(fout,";\n");
	}
	fprintf(fout,"];\n");
    }

    fprintf(fout, "bounds = [%d, %d, %d; %d, %d, %d];\n",cube.bl.i, cube.bl.j, cube.bl.k, cube.ur.i, cube.ur.j, cube.ur.k);
    fclose(fout);

}

void ConstraintMap::drawValidConfigsSmall() {
    CellIndex id, id2;
    Eigen::Vector3f pt;
    //the non-selected valid configs in green...
    for (id.i=0; id.i<n_v; ++id.i) {
	for (id.j=0; id.j<n_o; ++id.j) {
	    for (id.k=0; id.k<n_d; ++id.k) {
		int config_index =  id.k + n_d*id.j + n_d*n_o*id.i;
		if(valid_configs[config_index] == NULL) continue;
		if(valid_configs[config_index]->isValid) {
		    pt = valid_configs[config_index]->pose.translation();
		    this->getIdxPoint(pt,id2);
		    this->setFree(id2);
		}
	    }
	}
    }

    for(id.i = cube.bl.i; id.i<=cube.ur.i; ++id.i) {
	for(id.j = cube.bl.j; id.j<=cube.ur.j; ++id.j) {
	    for(id.k = cube.bl.k; id.k<=cube.ur.k; ++id.k) { 
		int config_index =  id.k + n_d*id.j + n_d*n_o*id.i;
		//std::cerr<<config_index<<" "<<valid_configs.size()<<std::endl;
		if(valid_configs[config_index] == NULL) {
		    std::cerr<<"there is an invalid config at "<<id.i<<","<<id.j<<","<<id.k<<std::endl;
		    continue;
		}
		if(valid_configs[config_index]->isValid) {
		    pt = valid_configs[config_index]->pose.translation();
		    this->getIdxPoint(pt,id2);
		    this->setOccupied(id2);
		    //std::cerr<<"set valid config at "<<id.i<<","<<id.j<<","<<id.k<<std::endl;
		} else {
		    std::cerr<<"there is an invalid config at "<<id.i<<","<<id.j<<","<<id.k<<std::endl;
		}
	    }
	}
    }
}
	
bool ConstraintMap::getPoseForConfig(CellIndex &id, Eigen::Affine3f &pose) {

    if(id.i<0 || id.i >= n_v) return false;
    if(id.j<0 || id.j >= n_o) return false;
    if(id.k<0 || id.k >= n_d) return false;

    if(!isSphereGrid) {
	pose = Eigen::AngleAxisf(2*id.j*M_PI/n_o, Eigen::Vector3f::UnitZ());
	Eigen::Vector3f ori = Eigen::Vector3f::UnitY();
	ori = pose*ori;
	ori *=(min_dist + ((max_dist-min_dist)*id.k)/n_d);
	ori(2) = min_z + ((max_z-min_z)*id.i)/n_v;
	pose = Eigen::AngleAxisf(M_PI + 2*id.j*M_PI/n_o, Eigen::Vector3f::UnitZ());
	pose.translation() = ori;
    } else {
	float theta, phi, r;
	theta = id.i*M_PI/n_v;
	phi = 2*id.j*M_PI/n_o;
	r = (min_dist + ((max_dist-min_dist)*id.k)/n_d);
	pose = Eigen::AngleAxisf(M_PI/2+phi, Eigen::Vector3f::UnitZ())*Eigen::AngleAxisf((3*M_PI/2+theta), Eigen::Vector3f::UnitX());
	pose.translation()(0) = r*sinf(theta)*cosf(phi);
	pose.translation()(1) = r*sinf(theta)*sinf(phi);
	pose.translation()(2) = r*cosf(theta);
    }

}

void ConstraintMap::sampleGripperGridSphere(int n_vert_angles, int n_orient_angles, int n_dist_samples,
	float min_dist_, float max_dist_) {
    
    isSphereGrid=true;
    n_v = n_vert_angles;
    n_o = n_orient_angles;
    n_d = n_dist_samples;
    min_z = 0;
    max_z = 0;
    min_dist = min_dist_;
    max_dist = max_dist_;
    //generate new configs and add them to ValidConfigs 
    
    float theta, phi, r;
    for(int i=0; i<n_v; ++i) {
	theta = i*M_PI/n_v;
	for(int j=0; j<n_o; ++j) {
	    phi = 2*j*M_PI/n_o;
	    for(int k=0; k<n_d; ++k) {
		r = (min_dist + ((max_dist-min_dist)*k)/n_d);
		Eigen::Affine3f pose;
		//pose = Eigen::AngleAxisf(M_PI-2*j*M_PI/n_o, Eigen::Vector3f::UnitZ())*Eigen::AngleAxisf(M_PI-2*i*M_PI/n_v, Eigen::Vector3f::UnitX());
		pose = Eigen::AngleAxisf(M_PI/2+phi, Eigen::Vector3f::UnitZ())*Eigen::AngleAxisf((3*M_PI/2+theta), Eigen::Vector3f::UnitX());
		//pose.setIdentity();
		pose.translation()(0) = r*sinf(theta)*cosf(phi);
		pose.translation()(1) = r*sinf(theta)*sinf(phi);
		pose.translation()(2) = r*cosf(theta);

		if(!isPJGripper && model != NULL) {
		    GripperConfiguration *config = new GripperConfiguration(pose,model);
		    config->calculateConstraints();
		    valid_configs.push_back(config);
		}
		if(isPJGripper && modelPJ!=NULL) {
		    GripperConfigurationPJ *config = new GripperConfigurationPJ(pose,modelPJ);
		    config->calculateConstraints();
		    valid_configs.push_back(config);
		}
		//valid_configs.size () = k + j*n_dist_samples + i*n_orient*n_dist_samples
    	    }
	}
    }

}

void ConstraintMap::sampleGripperGrid(int n_vert_slices, int n_orient, int n_dist_samples,
		float min_z_, float max_z_, float min_dist_, float max_dist_) {

    n_v = n_vert_slices;
    n_o = n_orient;
    n_d = n_dist_samples;
    min_z = min_z_;
    max_z = max_z_;
    min_dist = min_dist_;
    max_dist = max_dist_;

    //generate new configs and add them to ValidConfigs 
    for(int i=0; i<n_vert_slices; ++i) {
	for(int j=0; j<n_orient; ++j) {
	    for(int k=0; k<n_dist_samples; ++k) {
		//z pos
		float z = min_z + ((max_z-min_z)*i)/n_vert_slices;
		Eigen::Affine3f pose;
		pose = Eigen::AngleAxisf(2*j*M_PI/n_orient, Eigen::Vector3f::UnitZ());
		Eigen::Vector3f ori = Eigen::Vector3f::UnitY();
		ori = pose*ori;
		ori *=(min_dist + ((max_dist-min_dist)*k)/n_dist_samples);
		ori(2) = z;
		pose = Eigen::AngleAxisf(M_PI + 2*j*M_PI/n_orient, Eigen::Vector3f::UnitZ());
		pose.translation() = ori;

		if(!isPJGripper && model != NULL) {
		    GripperConfiguration *config = new GripperConfiguration(pose,model);
		    config->calculateConstraints();
		    valid_configs.push_back(config);
		}
		if(isPJGripper && modelPJ!=NULL) {
		    GripperConfigurationPJ *config = new GripperConfigurationPJ(pose,modelPJ);
		    config->calculateConstraints();
		    valid_configs.push_back(config);
		}
		//valid_configs.size () = k + j*n_dist_samples + i*n_orient*n_dist_samples
    	    }
	}
    }
}
	
void ConstraintMap::computeValidConfigs(SimpleOccMapIfce *object_map, Eigen::Affine3f cpose, float cradius, float cheight, 
		    Eigen::Affine3f &prototype_orientation, double orientation_tolerance, GripperPoseConstraint &output) {

    CylinderConstraint cylinder(cpose,cradius,cheight);
    Eigen::Vector3f tmp = cpose.translation();
    SphereConstraint sphere(tmp,cradius);
    double t1 = getDoubleTime();
    Eigen::AngleAxisf aa;
    float gyaw = prototype_orientation.rotation().eulerAngles(0,1,2)(2)+M_PI/2;
    //float gyaw = prototype_orientation.rotation().eulerAngles(0,1,2)(2);
    float MIN_OA,MAX_OA;
    if(isPJGripper) {
	MIN_OA = modelPJ->min_oa;
	MAX_OA = modelPJ->max_oa;
    } else {
	MIN_OA = model->min_oa;
	MAX_OA = model->max_oa;
    }
    //reset grid to make all configs valid
    for(int i=0; i<valid_configs.size(); ++i) {
	valid_configs[i]->isValid = true;
	valid_configs[i]->min_oa = MIN_OA; 
	valid_configs[i]->max_oa = MAX_OA;
	//EXCEPT the ones with a wrong orientation...
	float cyaw = valid_configs[i]->pose.rotation().eulerAngles(0,1,2)(2);
	float diff = fabs(gyaw-cyaw);
	if(diff > 2*M_PI) diff -= 2*M_PI; //one period over  
	//if(diff > M_PI) diff -= M_PI;	  //treat orientations equally in both directions
	//note orientation_tolerance < 0 => disable orientation bias
	valid_configs[i]->isValid = (orientation_tolerance < 0) || ( diff <= orientation_tolerance); 
    }
    
    CellIndex id;
    
    std::vector<CellIndex> overlap;
    this->getIntersectionWithPose(object_map,cylinder.pose,overlap);
    
    Eigen::Vector3f x, x_map;
    std::cerr<<"overlap size is "<<overlap.size()<<std::endl;
    for(int i=0; i<overlap.size(); ++i) {
	id = overlap[i];
        this->getCenterCell(id,x);
	x_map = cylinder.pose*x;	
	for(int j=0; j<config_grid[id.i][id.j][id.k].configs.size(); ++j) {
	    if(config_grid[id.i][id.j][id.k].configs[j]!=NULL) {
		if(*config_grid[id.i][id.j][id.k].configs[j]!=NULL) {
		    if((*config_grid[id.i][id.j][id.k].configs[j])->isValid) {
			if(!isSphereGrid) {
			    //check if x inside cylinder
			    if(cylinder(x_map)) {
				//if yes, update min_oa
				(*config_grid[id.i][id.j][id.k].configs[j])->updateMinAngle(x);
			    } else {
				//if no, update max_oa
				(*config_grid[id.i][id.j][id.k].configs[j])->updateMaxAngle(x);
			    }
			    if((*config_grid[id.i][id.j][id.k].configs[j])->max_oa <= (*config_grid[id.i][id.j][id.k].configs[j])->min_oa ||
				    (*config_grid[id.i][id.j][id.k].configs[j])->palm(x) ) {
				//if max_oa < min_oa, or if intersect with palm, -> config is not valid delete config
				(*config_grid[id.i][id.j][id.k].configs[j])->isValid = false;
			    } 
			} else {
			    //same check but with a sphere
			    if(sphere(x_map)) {
				//if yes, update min_oa
				(*config_grid[id.i][id.j][id.k].configs[j])->updateMinAngle(x);
			    } else {
				//if no, update max_oa
				(*config_grid[id.i][id.j][id.k].configs[j])->updateMaxAngle(x);
			    }
			    if((*config_grid[id.i][id.j][id.k].configs[j])->max_oa <= (*config_grid[id.i][id.j][id.k].configs[j])->min_oa ||
				    (*config_grid[id.i][id.j][id.k].configs[j])->palm(x) ) {
				//if max_oa < min_oa, or if intersect with palm, -> config is not valid delete config
				(*config_grid[id.i][id.j][id.k].configs[j])->isValid = false;
			    }
			}
		    }
		}
	    }
	}
    }

    //std::cerr<<n_v<<" "<<n_o<<" "<<n_d<<std::endl;
    if(config_sample_grid != NULL) {
	delete config_sample_grid;
    }
    config_sample_grid = new SimpleOccMap(0,0,0,1,n_v,n_o,n_d);
    config_sample_grid->setAllFree();
    
    std::vector<GripperConfiguration*> valid_configs2;
    for(int i=0; i<valid_configs.size(); ++i) {
	if(valid_configs[i]!=NULL) {
	    //check if we have something inside the gripper
	    if(valid_configs[i]->isValid && valid_configs[i]->min_oa > 0) {
		valid_configs2.push_back(valid_configs[i]);
	    } else {
		valid_configs[i]->isValid = false;
		//std::cerr<<"id "<<i<<" is "<<id.i<<","<<id.j<<","<<id.k<<" ?== "<<(i == id.k + n_d*id.j + n_d*n_o*id.i)<<std::endl;
		CellIndex id;
		id.k = i%n_d;
		id.j = ((i-id.k)/n_d)%n_o;
		id.i = (i-id.k-id.j*n_d)/(n_o*n_d);
		config_sample_grid->setOccupied(id);
		//delete valid_configs[i];
		//valid_configs[i] = NULL;
	    }
	} else {
	    std::cerr<<"Config is null, should not happen!\n";
	    exit(-1);
	}	    
    }
    double t1i = getDoubleTime();
    DfunMaxEmptyCubeExtractor extractor;
    extractor.loopY = true;
    if(isSphereGrid) extractor.loopX = true;
    //MaxEmptyCubeExtractor extractor;
    cube = extractor.getMaxCube2(config_sample_grid);

    double t2 = getDoubleTime();
    std::cerr<<"Had "<<valid_configs.size()<<" configs, now we have "<<valid_configs2.size()<<" and it took "<<t2-t1<<" seconds\n";
    std::cout<<"extract took :"<<t2-t1i<<" sec\n";
    int xlen = (cube.ur.i+n_v)%n_v - (cube.bl.i+n_v)%n_v+1;
    int ylen = (cube.ur.j+n_o)%n_o - (cube.bl.j+n_o)%n_o+1;
    int volume = (cube.ur.k - cube.bl.k+1)*xlen*ylen;
    std::cout<<"MAX cube at ("<<cube.bl.i<<","<<cube.bl.j<<","<<cube.bl.k<<") : ("<<cube.ur.i<<","<<cube.ur.j<<","<<cube.ur.k<<") size: "<<xlen<<","<<ylen
	     <<" volume "<<volume<<" cvolume "<<cube.volume()<<std::endl;

    output.max_oa = M_PI;
    output.min_oa = 0;

    CellIndex id2;
    int v=0;
    int dctr=0;
    for(id.i = cube.bl.i; id.i<=cube.ur.i; ++id.i) {
	for(id.j = cube.bl.j; id.j<=cube.ur.j; ++id.j) {
	    for(id.k = cube.bl.k; id.k<=cube.ur.k; ++id.k) {
		//id2.i = (id.i + map_size)%map_size;
		//id2.j = (id.j + map_size)%map_size;
		//id2.k = (id.k + map_size)%map_size;
		v = id.i*n_o*n_d + id.j*n_d + id.k;
		if(valid_configs[v] !=NULL) {
		    if(valid_configs[v]->isValid) {
			//std::cout<<v<<" "<<id.i<<" "<<id.j<<" "<<id.k<<" "<<valid_configs[v]->min_oa<<" "<<valid_configs[v]->max_oa<<std::endl;
			output.min_oa = valid_configs[v]->min_oa > output.min_oa ? valid_configs[v]->min_oa : output.min_oa;
			output.max_oa = valid_configs[v]->max_oa < output.max_oa ? valid_configs[v]->max_oa : output.max_oa;
		    }
		}
	    
		dctr++;
	    }
	}
    }
    //This converts to the scale expected by the velvet gripper
    //output.max_oa = fabsf((M_PI-output.max_oa)/2);
    //output.min_oa = (M_PI-output.min_oa)/2;
    std::cout<<"min "<<output.min_oa<<" max "<<output.max_oa<<" checked "<<dctr<<std::endl;
    output.debug_time = t2-t1;
    output.cspace_volume = cube.volume();

    if(!isSphereGrid) {
	output.isSphere = false;
	
	//bottom left gives bottom plane, left plane and outer cylinder
	Eigen::Affine3f pose;
	pose = Eigen::AngleAxisf(2*cube.bl.j*M_PI/n_o, Eigen::Vector3f::UnitZ());
	Eigen::Vector3f ori = Eigen::Vector3f::UnitY();
	ori = pose*ori;
	
	output.lower_plane.a = Eigen::Vector3f::UnitZ();
	output.lower_plane.b = min_z + ((max_z-min_z)*cube.bl.i)/n_v;
	output.left_bound_plane.a = ori.cross(Eigen::Vector3f::UnitZ());
	output.left_bound_plane.b = 0;
	pose.setIdentity();
	output.inner_cylinder = CylinderConstraint(pose, (min_dist + ((max_dist-min_dist)*cube.bl.k)/n_d), max_z-min_z);

	//upper right gives upper plane, right plane and inner cylinder
	pose = Eigen::AngleAxisf(2*cube.ur.j*M_PI/n_o, Eigen::Vector3f::UnitZ());
	ori = pose*Eigen::Vector3f::UnitY();

	output.upper_plane.a = Eigen::Vector3f::UnitZ();
	output.upper_plane.b = min_z + ((max_z-min_z)*cube.ur.i)/n_v;
	output.right_bound_plane.a = ori.cross(Eigen::Vector3f::UnitZ());
	output.right_bound_plane.b = 0;
	pose.setIdentity();
	output.outer_cylinder = CylinderConstraint(pose, (min_dist + ((max_dist-min_dist)*cube.ur.k)/n_d), max_z-min_z);
    } else {
	output.isSphere = true;
	float theta, phi, r, phi_midpoint;
	Eigen::Affine3f pose;
	Eigen::Vector3f ori;

	//bottom left gives bottom plane, left plane and outer sphere
	theta = cube.bl.i*M_PI/n_v;
	phi = 2*cube.bl.j*M_PI/n_o;
	phi_midpoint = (cube.bl.j+cube.ur.j)*M_PI/n_o;
	r = (min_dist + ((max_dist-min_dist)*cube.bl.k)/n_d);

	/* //Note: this is for angle-aware slices
	pose = Eigen::AngleAxisf(M_PI/2+phi_midpoint, Eigen::Vector3f::UnitZ())*Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitX());
	ori = pose*Eigen::Vector3f::UnitY();
	output.lower_plane.a = ori;
	output.lower_plane.b = 0;
	*/
	//vertical slices
	output.lower_plane.a = Eigen::Vector3f::UnitZ();
	output.lower_plane.b = r*cosf(theta);

	pose = Eigen::AngleAxisf(phi, Eigen::Vector3f::UnitZ());
	ori = pose*Eigen::Vector3f::UnitY();
	output.left_bound_plane.a = ori;
	output.left_bound_plane.b = 0;

	ori.setZero();
	output.outer_sphere.center = ori;
	output.outer_sphere.radius = r;

	//upper right gives upper plane, right plane and inner cylinder
	theta = cube.ur.i*M_PI/n_v;
	phi = 2*cube.ur.j*M_PI/n_o;
	r = (min_dist + ((max_dist-min_dist)*cube.ur.k)/n_d);

	/* //Note: this is for angle-aware slices
	pose = Eigen::AngleAxisf(M_PI/2+phi_midpoint, Eigen::Vector3f::UnitZ())*Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitX());
	ori = pose*Eigen::Vector3f::UnitY();
	output.upper_plane.a = ori;
	output.upper_plane.b = 0;
	*/
	//vertical slices
	output.upper_plane.a = Eigen::Vector3f::UnitZ();
	output.upper_plane.b = r*cosf(theta);
	
	pose = Eigen::AngleAxisf(phi, Eigen::Vector3f::UnitZ());
	ori = pose*Eigen::Vector3f::UnitY();
	output.right_bound_plane.a = ori;
	output.right_bound_plane.b = 0;

	ori.setZero();
	output.inner_sphere.center = ori;
	output.inner_sphere.radius = r;

    }
//    valid_configs = valid_configs2;
}
	
bool ConstraintMap::saveGripperConstraints(const char *fname) const {
    //what to save:
    FILE *fout = fopen(fname, "w+b");
    if(fout == NULL) {
	std::cerr<<"couldn't open file for writing at "<<fname<<std::endl;
	return false;
    }
    std::cerr<<"writing "<<_FILE_VERSION_<<std::endl;
    fwrite(_FILE_VERSION_, sizeof(char), strlen(_FILE_VERSION_), fout);
    //center;
    std::cerr<<"writing center\n";
    fwrite(center.data(), sizeof(float), 3, fout);
    //size_meters;
    fwrite(size_meters.data(), sizeof(float), 3, fout);
    //resolution;
    fwrite(&resolution, sizeof(float), 1, fout);
	
    //grid;
    CellIndex id;
    for (id.i=0; id.i<size_x; ++id.i) {
	for (id.j=0; id.j<size_y; ++id.j) {
	    fwrite(grid[id.i][id.j], sizeof(int), size_z, fout);
	}
    }
   
    fwrite(&isPJGripper, sizeof(bool), 1, fout);
    if(!isPJGripper && model != NULL) {  
	//GripperModel *model;
	fwrite(model->finger_size.data(), sizeof(float), 3, fout);
	fwrite(model->palm_size.data(), sizeof(float), 3, fout);
	saveMatrix4f(model->palm2left.matrix(), fout);
	saveMatrix4f(model->palm2right.matrix(), fout);
	saveMatrix4f(model->palm2fingers.matrix(), fout);
	saveMatrix4f(model->palm2palm_box.matrix(), fout);
	saveMatrix4f(model->left2left_box.matrix(), fout);
	saveMatrix4f(model->right2right_box.matrix(), fout);
    } 
    else if (isPJGripper && modelPJ != NULL) {
	fwrite(modelPJ->finger_size.data(), sizeof(float), 3, fout);
	fwrite(modelPJ->palm_size.data(), sizeof(float), 3, fout);
	saveMatrix4f(modelPJ->palm2palm_box.matrix(), fout);
	saveMatrix4f(modelPJ->palm2fingers_box.matrix(), fout);
	fwrite(&modelPJ->max_oa, sizeof(float), 1, fout);
	fwrite(&modelPJ->finger_thickness, sizeof(float), 1, fout);
    }

    //std::vector<GripperConfiguration*> valid_configs;
    size_t sz1 =valid_configs.size(); 
    fwrite(&sz1, sizeof(size_t), 1, fout);
    for(int i=0; i<valid_configs.size(); ++i) {
	saveMatrix4f(valid_configs[i]->pose.matrix(), fout);
    }
	
    //ConfigurationList *** config_grid;
    for (id.i=0; id.i<size_x; ++id.i) {
	for (id.j=0; id.j<size_y; ++id.j) {
	    for (id.k=0; id.k<size_z; ++id.k) {
		size_t sz =config_grid[id.i][id.j][id.k].config_ids.size(); 
		fwrite(&sz, sizeof(size_t), 1, fout);
		fwrite(config_grid[id.i][id.j][id.k].config_ids.data(), sizeof(int), sz, fout);
	    }
	}
    }
    fwrite(&n_v, sizeof(int), 1, fout);
    fwrite(&n_o, sizeof(int), 1, fout);
    fwrite(&n_d, sizeof(int), 1, fout);
    
    fwrite(&min_z, sizeof(float), 1, fout);
    fwrite(&max_z, sizeof(float), 1, fout);
    fwrite(&min_dist, sizeof(float), 1, fout);
    fwrite(&max_dist, sizeof(float), 1, fout);
    
    fwrite(&isSphereGrid, sizeof(bool), 1, fout);

    fclose(fout);
    return true;
}

bool ConstraintMap::loadGripperConstraints(const char *fname) {
    FILE *fin = fopen(fname, "r+b");
    if(fin == NULL) {
	std::cerr<<"couldn't open file for reading at "<<fname<<std::endl;
	return false;
    }
    
    char versionBuf[16];
    if(fread(&versionBuf, sizeof(char), strlen(_FILE_VERSION_), fin) <= 0)
    {
	std::cerr<<"reading version failed\n";
	return false;
    }
    versionBuf[strlen(_FILE_VERSION_)] = '\0';
    bool isOldFileVersion = false;
    if(strncmp(versionBuf, _FILE_VERSION_, strlen(_FILE_VERSION_)) != 0) {
	if(strncmp(versionBuf, _FILE_VERSION_IROS2016, strlen(_FILE_VERSION_IROS2016)) != 0) {
	    std::cerr<<"Version mismatch, don't know what to do: "<<versionBuf<<std::endl;
	    return false;
	} else {
	    isOldFileVersion = true;
	}
    }
    
    //center;
    if(fread(center.data(), sizeof(float), 3, fin)!=3) {
	std::cerr<<"couldn't read\n";
	return false;
    }
    //size_meters;
    if(fread(size_meters.data(), sizeof(float), 3, fin)!=3) {
	std::cerr<<"couldn't read\n";
	return false;
    }
    //resolution;
    if(fread(&resolution, sizeof(float), 1, fin)!=1) {
	std::cerr<<"couldn't read\n";
	return false;
    }
    size_x = size_meters(0) / resolution;
    size_y = size_meters(1) / resolution;
    size_z = size_meters(2) / resolution;

    std::cerr<<"read c "<<center.transpose()<<" s "<<size_meters.transpose()<<" r "<<resolution<<std::endl;
    std::cerr<<"collision map size is "<<size_x<<" "<<size_y<<" "<<size_z<<" cells\n";

    if(!isInitialized) initialize();
    initializeConfigs();
    
    std::cerr<<"reading collision map ...\n";
    //grid;
    CellIndex id;
    for (id.i=0; id.i<size_x; ++id.i) {
	for (id.j=0; id.j<size_y; ++id.j) {
	    if(fread(grid[id.i][id.j], sizeof(int), size_z, fin)!=size_z) {
		std::cerr<<"couldn't read grid\n";
		return false;
	    }
	}
    }
    std::cerr<<"DONE\n";
    //sleep(5);
    
    if(isOldFileVersion) { 
	model = new GripperModel();
	isPJGripper = false;
	modelPJ = NULL;
    } 
    else {
	//read in gripper type
	if(fread(&isPJGripper, sizeof(bool), 1, fin) != 1) {
	    std::cerr<<"couldn't read in gripper type\n";
	    return false;
	}
        if(isPJGripper) {
	    modelPJ = new GripperModelPJ();
	    model = NULL;
	    std::cout<<"read a PJ gripper type\n";
	} else {
	    model = new GripperModel();
	    modelPJ = NULL;
	    std::cout<<"read a Velvet gripper type\n";
	}	    
    }
    hasGripper = true; 

    if(isOldFileVersion || !isPJGripper) {
	if(fread(model->finger_size.data(), sizeof(float), 3, fin)!=3) {
	    std::cerr<<"couldn't read\n";
	    return false;
	}
	if(fread(model->palm_size.data(), sizeof(float), 3, fin)!=3) {
	    std::cerr<<"couldn't read\n";
	    return false;
	}
	if(!readMatrix4f(model->palm2left.matrix(), fin)) return false;
	if(!readMatrix4f(model->palm2right.matrix(), fin)) return false;
	if(!readMatrix4f(model->palm2fingers.matrix(), fin)) return false;
	if(!readMatrix4f(model->palm2palm_box.matrix(), fin)) return false;
	if(!readMatrix4f(model->left2left_box.matrix(), fin)) return false;
	if(!readMatrix4f(model->right2right_box.matrix(), fin)) return false;
    } 
    else {
	if(fread(modelPJ->finger_size.data(), sizeof(float), 3, fin)!=3) {
	    std::cerr<<"couldn't read finger dimensions\n";
	    return false;
	}
	if(fread(modelPJ->palm_size.data(), sizeof(float), 3, fin)!=3) {
	    std::cerr<<"couldn't read palm dimensions\n";
	    return false;
	}
	if(!readMatrix4f(modelPJ->palm2palm_box.matrix(), fin)) return false;
	if(!readMatrix4f(modelPJ->palm2fingers_box.matrix(), fin)) return false;
	if(fread(&modelPJ->max_oa, sizeof(float), 1, fin)!=1) {
	    std::cerr<<"couldn't read gripper opening size\n";
	    return false;
	}
	if(fread(&modelPJ->finger_thickness, sizeof(float), 1, fin)!=1) {
	    std::cerr<<"couldn't read gripper finger thickness\n";
	    return false;
	}
    }
    
    size_t sz1;
    if(fread(&sz1, sizeof(size_t), 1, fin)!=1) {
	std::cerr<<"couldn't read\n";
	return false;
    }
    std::cerr<<"reading configuration grid and initializing configs vector...\n";
    for(int i=0; i<sz1; ++i) {
	Eigen::Affine3f ps;
	if(!readMatrix4f(ps.matrix(), fin)) return false;
	if(!isPJGripper && model!=NULL) { 
	    GripperConfiguration *cn = new GripperConfiguration(ps,model);
	    cn->calculateConstraints();
	    valid_configs.push_back(cn);
	} 
	else if(isPJGripper && modelPJ!=NULL) {
	    GripperConfigurationPJ *cn = new GripperConfigurationPJ(ps,modelPJ);
	    cn->calculateConstraints();
	    valid_configs.push_back(cn);
	} else {
	    std::cerr<<"could not get gripper model\n";
	    return false;
	}
    }
    std::cerr<<"DONE\n";
    //sleep(5);

    std::cerr<<"initializing/reading configuration grid ...\n";
    for (id.i=0; id.i<size_x; ++id.i) {
	for (id.j=0; id.j<size_y; ++id.j) {
	    for (id.k=0; id.k<size_z; ++id.k) {
		size_t sz;
		if(fread(&sz, sizeof(size_t), 1, fin)!=1) {
		    std::cerr<<"couldn't read\n";
		    return false;
		}
		int idx;
		for(int i=0; i<sz; ++i) {
		    if(fread(&idx, sizeof(int), 1, fin)!=1) {
			std::cerr<<"couldn't read\n";
			return false;
		    }
		    if(idx <0 || idx > valid_configs.size()) {
			std::cerr<<"something is wrong, config id out of bounds "<<idx<<std::endl;
			return false;
		    }
		    //config_grid[id.i][id.j][id.k].config_ids.push_back(idx);
		    config_grid[id.i][id.j][id.k].configs.push_back(&valid_configs[idx]);
		}
	    }
	}
    }
    std::cerr<<"DONE\n";
    //sleep(5);
		
    if(fread(&n_v, sizeof(int), 1, fin)!=1) {
	std::cerr<<"couldn't read\n";
	return false;
    }
    if(fread(&n_o, sizeof(int), 1, fin)!=1) {
	std::cerr<<"couldn't read\n";
	return false;
    }
    if(fread(&n_d, sizeof(int), 1, fin)!=1) {
	std::cerr<<"couldn't read\n";
	return false;
    }
    
    if(fread(&min_z, sizeof(float), 1, fin)!=1) {
	std::cerr<<"couldn't read\n";
	return false;
    }
    if(fread(&max_z, sizeof(float), 1, fin)!=1) {
	std::cerr<<"couldn't read\n";
	return false;
    }
    if(fread(&min_dist, sizeof(float), 1, fin)!=1) {
	std::cerr<<"couldn't read\n";
	return false;
    }
    if(fread(&max_dist, sizeof(float), 1, fin)!=1) {
	std::cerr<<"couldn't read\n";
	return false;
    }
    
    if(fread(&isSphereGrid, sizeof(bool), 1, fin)!=1) {
	std::cerr<<"couldn't read, assume cylinder grid\n";
	isSphereGrid=false;
    }
    fclose(fin);
    std::cerr<<"DONE loading file\n";

    return true;
}
