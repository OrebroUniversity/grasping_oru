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
    
    double t1 = getDoubleTime();
    std::vector<GripperConfigurationSimple> simple_configs;
    for(int i=0; i<valid_configs.size(); ++i) {
	if(valid_configs[i] == NULL) continue;
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
	
void ConstraintMap::drawValidConfigsSmall() {
    CellIndex id, id2;
    Eigen::Vector3f pt;
    //the non-selected valid configs in green...
    for (id.i=0; id.i<n_v; ++id.i) {
	for (id.j=0; id.j<n_o; ++id.j) {
	    for (id.k=0; id.k<n_d; ++id.k) {
		int config_index =  id.k + n_d*id.j + n_d*n_o*id.i;
		if(valid_configs[config_index] != NULL) {
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
		if(valid_configs[config_index] != NULL) {
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
	
void ConstraintMap::sampleGripperGrid(int n_vert_slices, int n_orient, int n_dist_samples,
		float min_z, float max_z, float min_dist, float max_dist) {

    n_v = n_vert_slices;
    n_o = n_orient;
    n_d = n_dist_samples;

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

		GripperConfiguration *config = new GripperConfiguration(pose,model);
		config->calculateConstraints();
		valid_configs.push_back(config);
		//valid_configs.size () = k + j*n_dist_samples + i*n_orient*n_dist_samples
    	    }
	}
    }
}
	
void ConstraintMap::computeValidConfigs(SimpleOccMapIfce *object_map, CylinderConstraint &cylinder) {

    double t1 = getDoubleTime();
    //pass through valid_configs and remove the crazy ones
/*    for(int i=0; i<valid_configs.size(); ++i) {
	
    }
    */
    std::vector<CellIndex> overlap;
    this->getIntersectionWithPose(object_map,cylinder.pose,overlap);
    
    Eigen::Vector3f x, x_map;
    CellIndex id;
    std::cerr<<"overlap size is "<<overlap.size()<<std::endl;
    for(int i=0; i<overlap.size(); ++i) {
	id = overlap[i];
        this->getCenterCell(id,x);
	x_map = cylinder.pose*x;	
	for(int j=0; j<config_grid[id.i][id.j][id.k].configs.size(); ++j) {
	    if(config_grid[id.i][id.j][id.k].configs[j]!=NULL) {
		if(*config_grid[id.i][id.j][id.k].configs[j]!=NULL) {
		    //check if x inside cylinder
		    if(cylinder(x_map)) {
			//if yes, update min_oa
			(*config_grid[id.i][id.j][id.k].configs[j])->updateMinAngle(x);
		    } else {
			/*std::cerr<<"collision outside cylinder at "<<x.transpose()<<std::endl;
			CellIndex id2;
			object_map->getIdxPoint(x,id2);
			object_map->setFree(id2);

			Eigen::Matrix<float,2,1> bp = cylinder.A*x-cylinder.b;
			if(bp(0)<0) std::cerr<<"bp fault\n";
			if(bp(1)<0) std::cerr<<"bp fault\n";
			Eigen::Vector3f normal = cylinder.A.block<1,3>(0,0).transpose();
			Eigen::Vector3f xt = x - cylinder.pose.translation();
			Eigen::Vector3f rejection = xt - xt.dot(normal)*normal;
			if(rejection.norm() > cylinder.radius_) std::cerr<<"radius fault "<<rejection.norm()<<std::endl;*/
			//if no, update max_oa
			(*config_grid[id.i][id.j][id.k].configs[j])->updateMaxAngle(x);
		    }
		    if((*config_grid[id.i][id.j][id.k].configs[j])->max_oa <= (*config_grid[id.i][id.j][id.k].configs[j])->min_oa ||
			   (*config_grid[id.i][id.j][id.k].configs[j])->palm(x) ) {
			//if max_oa < min_oa, delete config
			delete *config_grid[id.i][id.j][id.k].configs[j];
			*config_grid[id.i][id.j][id.k].configs[j] = NULL;
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
	    if(valid_configs[i]->min_oa > 0) {
		valid_configs2.push_back(valid_configs[i]);
	    } else {
		delete valid_configs[i];
		valid_configs[i] = NULL;
	    }
	}	    
	//update config_sample_grid
	if(valid_configs[i]==NULL) {
	    CellIndex id;
	    id.k = i%n_d;
	    id.j = ((i-id.k)/n_d)%n_o;
	    id.i = (i-id.k-id.j*n_d)/(n_o*n_d);
	    config_sample_grid->setOccupied(id);
	    //std::cerr<<"id "<<i<<" is "<<id.i<<","<<id.j<<","<<id.k<<" ?== "<<(i == id.k + n_d*id.j + n_d*n_o*id.i)<<std::endl;
	}
    }
    double t1i = getDoubleTime();
    DfunMaxEmptyCubeExtractor extractor;
    //MaxEmptyCubeExtractor extractor;
    cube = extractor.getMaxCube(config_sample_grid);

    double t2 = getDoubleTime();
    std::cerr<<"Had "<<valid_configs.size()<<" configs, now we have "<<valid_configs2.size()<<" and it took "<<t2-t1<<" seconds\n";
    std::cout<<"extract took :"<<t2-t1i<<" sec\n";
    std::cout<<"MAX cube at ("<<cube.bl.i<<","<<cube.bl.j<<","<<cube.bl.k<<") : ("<<cube.ur.i<<","<<cube.ur.j<<","<<cube.ur.k<<") volume "<<cube.volume()<<std::endl;

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
    
    //GripperModel *model;
    fwrite(model->finger_size.data(), sizeof(float), 3, fout);
    fwrite(model->palm_size.data(), sizeof(float), 3, fout);
    saveMatrix4f(model->palm2left.matrix(), fout);
    saveMatrix4f(model->palm2right.matrix(), fout);
    saveMatrix4f(model->palm2fingers.matrix(), fout);
    saveMatrix4f(model->palm2palm_box.matrix(), fout);
    saveMatrix4f(model->left2left_box.matrix(), fout);
    saveMatrix4f(model->right2right_box.matrix(), fout);

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
    if(strncmp(versionBuf, _FILE_VERSION_, strlen(_FILE_VERSION_)) != 0) {
	std::cerr<<"Version mismatch, don't know what to do: "<<versionBuf<<std::endl;
	return false;
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

    if(!isInitialized) initialize();
    model = new GripperModel();
    hasGripper = true; 
    initializeConfigs();
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
    
    size_t sz1;
    if(fread(&sz1, sizeof(size_t), 1, fin)!=1) {
	std::cerr<<"couldn't read\n";
	return false;
    }
    for(int i=0; i<sz1; ++i) {
	Eigen::Affine3f ps;
	if(!readMatrix4f(ps.matrix(), fin)) return false;
	GripperConfiguration *cn = new GripperConfiguration(ps,model);
	cn->calculateConstraints();
	valid_configs.push_back(cn);
    }

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
		    config_grid[id.i][id.j][id.k].config_ids.push_back(idx);
		    config_grid[id.i][id.j][id.k].configs.push_back(&valid_configs[idx]);
		}
	    }
	}
    }
		
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
    fclose(fin);

    return true;
}
