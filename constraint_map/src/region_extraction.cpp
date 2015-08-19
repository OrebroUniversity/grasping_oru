#include <constraint_map/RegionExtraction.hh>
#include <algorithm>

SegmentTree::SegmentTree (int minx_, int maxx_, int miny_, int maxy_, int xdir_, int ydir_) {
    minx = minx_;
    maxx = maxx_;
    miny = miny_;
    maxy = maxy_;
    xdir = xdir_;
    ydir = ydir_;
    
}

bool SegmentTree::isActive (int &x, int &y) {
    //are we inside the quadrant?
    if(x > maxx || x < minx || y > maxy || y < miny) return false;
    //create node
    SegmentNode sn (x,y,xdir,ydir);
    //find upper bound on x
    boost::intrusive::rbtree<SegmentNode>::iterator upper_x = tree.upper_bound(sn);
    //we are adding a new node with an x bigger than any previous ones
    if(upper_x == tree.end()) {
	//std::cerr<<"couldn't find a point with higher x\n";
	return true;
    }
    //std::cerr<<"found a point with higher x\n";
    //if bound on y is not satisfied, then we are not active.
    if(upper_x->ydir*upper_x->y > ydir*y) {
	return false;
    }
    //boundary cases
    if(upper_x->y == maxy || upper_x->y == miny) {
	return false;
    }
    return true;
}

void SegmentTree::insertPoint(int &x, int &y) {

    //are we inside the quadrant?
    if(x > maxx || x < minx || y > maxy || y < miny) return;
    //create node
    SegmentNode *sn = new SegmentNode(x,y,xdir,ydir);
    //find upper bound on x
    boost::intrusive::rbtree<SegmentNode>::iterator upper_x = tree.upper_bound(*sn);
    //we are adding a new node with an x bigger than any previous ones
    if(upper_x == tree.end()) {
	//std::cerr<<"couldn't find a point with higher x\n";
	//check for the y bound inside the tree
	boost::intrusive::rbtree<SegmentNode>::iterator upper_y = tree.upper_bound(*sn, &SegmentYCmp); 
	//if there is no node with a bigger y, we insert
	if(upper_y == tree.end()) {
	    //std::cerr<<"couldn't find point with smaller y, inserting\n";
	    tree.insert_equal(*sn);
	    return;
	}
	//std::cerr<<"found a point with smaller y\n";
	//if(upper_y!=tree.begin()) upper_y --;
	
	tree.erase_and_dispose(upper_y,upper_x,delete_disposer());
	tree.insert_equal(*sn);
	return;
    }

    //std::cerr<<"found a point with higher x\n";
    //if bound on y is not satisfied, then we are not active.
    if(upper_x->ydir*upper_x->y > ydir*y) {
	//std::cerr<<"..but it is also with higher y, we are outside active\n";
	delete sn;
	return;
    }

    //find upper bound on y
    boost::intrusive::rbtree<SegmentNode>::iterator upper_y = tree.upper_bound(*sn, &SegmentYCmp); 
    if(upper_y == tree.end()) {
	//not really possible
	std::cerr<<"something is wrong\n";
	delete sn;
	return;
    }
    //std::cerr<<"found point with smaller y\n";

    //delete the range between xbound and ybound
    //if(upper_x!=tree.begin()) upper_x --;
    //if(upper_y!=tree.begin()) upper_y --;
    tree.erase_and_dispose(upper_y,upper_x,delete_disposer());

    //insert new segment
    tree.insert_equal(*sn);
}
	
ActiveSet::ActiveSet(int x_, int y_, int z_, int minx_, int maxx_, int miny_, int maxy_):x(x_),y(y_),minx(minx_),miny(miny_),maxx(maxx_),maxy(maxy_),z(z_) {
   bl = new SegmentTree(minx,x,miny,y,1,1); 
   br = new SegmentTree(x,maxx,miny,y,-1,1); 
   ul = new SegmentTree(minx,x,y,maxy,1,-1); 
   ur = new SegmentTree(x,maxx,y,maxy,-1,-1); 
}

bool ActiveSet::isActive (int &x_, int &y_) {
   if(x_ < x && y_ < y) return bl->isActive(x_,y_);
   if(x_ > x && y_ < y) return br->isActive(x_,y_);
   if(x_ < x && y_ > y) return ul->isActive(x_,y_);
   if(x_ > x && y_ > y) return ur->isActive(x_,y_);
   if(x_ == x) {
       if(y_ > y) return ul->isActive(x_,y_) && ur->isActive(x_,y_);
       if(y_ < y) return br->isActive(x_,y_) && bl->isActive(x_,y_);
   }
   if(y_ == y) {
       if(x_ > x) return br->isActive(x_,y_) && ur->isActive(x_,y_);
       if(x_ < x) return ul->isActive(x_,y_) && bl->isActive(x_,y_);
   }
   return false;
}

void ActiveSet::insertPoint(int &x_, int &y_) {
   if(x_ <= x && y_ <= y) bl->insertPoint(x_,y_);
   if(x_ >= x && y_ <= y) br->insertPoint(x_,y_);
   if(x_ <= x && y_ >= y) ul->insertPoint(x_,y_);
   if(x_ >= x && y_ >= y) ur->insertPoint(x_,y_);
}
	
//computes the max area rectangle in the active set and stores bottom left/ upper right corners. returns area.
int ActiveSet::maxRectangle(int &xblf, int &yblf, int &xurf, int &yurf) {
    int xbl, ybl, xur, yur, max_area=-1;
    //std::cout<<"\n//////////////////////////////////////////////////////////////\n";
    //Cases 1-4
    {
	boost::intrusive::rbtree<SegmentNode>::iterator blit = bl->tree.begin();
	boost::intrusive::rbtree<SegmentNode>::iterator blit_prev = bl->tree.begin();
	bool check_empty = (bl->tree.begin() == bl->tree.end());
	//choose both from BL
	while (blit_prev != bl->tree.end() || check_empty) {
	    if(!check_empty) {
		if(blit!=bl->tree.end()) {
		    ybl = blit->y;
		} else {
		    ybl = bl->miny;
		}
		if(blit_prev == blit) {
		    xbl = bl->minx;
		    blit++;
		} else {
		    xbl = blit_prev->x;
		    blit++;
		    blit_prev++;
		}
		if(!bl->isActive(xbl,ybl)) {
		    continue;
		}
	    } else {
		check_empty = false;
		xbl = bl->minx;
		ybl = bl->miny;
	    }
	    //std::cout<<"\n\nchecking bottom corner at ("<<xbl<<","<<ybl<<")\n";
	    SegmentNode sn(xbl,ybl,1,-1);
	    //now we check for x upper bound in ul, the corresponding point's y is our ponential yur
	    boost::intrusive::rbtree<SegmentNode>::iterator ulit = ul->tree.upper_bound(sn);
	    //if(ulit == ul->tree.begin()) {
	    //    yur = ul->miny;
	    //    //std::cout<<"ul y at tree_start\n";
	    //} 
	    //else 
	    if(ulit ==ul->tree.end()) {
		yur = ul->maxy;
		//std::cout<<"ul y at tree_end\n";
	    } else {
		yur = ulit->y;
	    }
	    //now we check for y upper bound in br, the corresponding point's x is our ponential xur
	    sn.xdir = -1; sn.ydir = 1;
	    boost::intrusive::rbtree<SegmentNode>::iterator brit = br->tree.upper_bound(sn, &SegmentYCmp);
	    //if(brit == br->tree.begin()) {
	    //	xur = br->maxx;
	    //	//std::cout<<"br x at tree_start\n";
	    //    } 
	    //    else 
	    if(brit ==br->tree.end()) {
		brit--;
		xur = brit->x;
		//std::cout<<"br x at tree_end\n";
	    } else {
		xur = brit->x;
	    }
	    //std::cout<<"check ("<<xur<<","<<yur<<")\n";
	    if(ur->isActive(xur,yur)) {
		//std::cout<<"active!\n";
		//the region is valid, compute area
		int area = abs(yur-ybl)*abs(xur-xbl);
		if(area > max_area) {
		    //std::cout<<"1-4: "<<area<<std::endl;
		    max_area=area; xblf = xbl; yblf = ybl; xurf=xur; yurf=yur;
		    //std::cout<<"CHOOSE: ("<<xbl<<","<<ybl<<") : ("<<xur<<","<<yur<<")\n";
		}
	    } else {
		//std::cout<<"not active, let's check all inside:\n";
		sn.x = xur; sn.y = yur; sn.xdir=-1; sn.ydir=-1;
		//region is actually bounded by ur corner, check the possible supports
		boost::intrusive::rbtree<SegmentNode>::iterator xur_it = ur->tree.upper_bound(sn);
		boost::intrusive::rbtree<SegmentNode>::iterator yur_it = ur->tree.upper_bound(sn, &SegmentYCmp);
		//find which one is first accoridng to x order 
		if(xur_it->x*xur_it->xdir > yur_it->x*yur_it->xdir && yur_it != ur->tree.end()) {
		    //switch them
		    boost::intrusive::rbtree<SegmentNode>::iterator tmp = xur_it;
		    xur_it = yur_it;
		    yur_it = tmp;
		}
		boost::intrusive::rbtree<SegmentNode>::iterator tmp = xur_it;
		boost::intrusive::rbtree<SegmentNode>::iterator tmp2;
		while(tmp!=yur_it) {
		    tmp2=tmp;
		    tmp++;
		    int txur=tmp2->x;
		    int tyur=tmp != ur->tree.end() ? tmp->y : ur->maxy;
		    txur = xur*ur->xdir > txur*ur->xdir ? xur : txur; 
		    tyur = yur*ur->ydir > tyur*ur->ydir ? yur : tyur; 
		    //std::cout<<"check ("<<txur<<","<<tyur<<")\n";
		    //double check to be on the safe side
		    if(!isActive(txur,tyur)) { 
			//std::cerr<<"messed up at ("<<txur<<","<<tyur<<")\n";
			continue;
		    }
		    if(!isActive(txur,ybl)) { 
			//std::cerr<<"messed up at ("<<txur<<","<<ybl<<")\n";
			continue;
		    }
		    if(!isActive(xbl,tyur)) { 
			//std::cerr<<"messed up at ("<<xbl<<","<<tyur<<")\n";
			continue;
		    }
		    //the region is valid, compute area
		    int area = abs(tyur-ybl)*abs(txur-xbl);
		    if(area > max_area) {
			//std::cout<<"1-4(2): "<<area<<std::endl;
			max_area=area; xblf = xbl; yblf = ybl; xurf=txur; yurf=tyur;
			//std::cout<<"CHOOSE: ("<<xbl<<","<<ybl<<") : ("<<xur<<","<<yur<<")\n";
		    }
		}
	    }
	    //std::cout<<"CASES 15/16\n\n\n";
	    //handle cases 15 and 16
	    boost::intrusive::rbtree<SegmentNode>::iterator it;
	    //case 15: keep xbl and look for the rest
	    sn.x = xbl; sn.xdir = 1; sn.ydir = -1;
	    it = ul->tree.upper_bound(sn);
	    if(it == ul->tree.end()) {
		yur = ul->maxy;
	    } else {
		yur = it->y;
	    }
	    if(ul->tree.begin() == ul->tree.end()) yur = ul->maxy;
	    //std::cout<<"got point ("<<it->x<<","<<it->y<<"), yur is "<<yur<<std::endl;
	    sn.y = yur; sn.xdir = -1;
	    it = ur->tree.upper_bound(sn, &SegmentYCmp);
	    if(it == ur->tree.begin()) {
		xur = ur->maxx;
	    } else {
		it--;
		xur = it->x;
	    }
	    if(ur->tree.begin() == ur->tree.end()) xur = ur->maxx;
	    //std::cout<<"got point ("<<it->x<<","<<it->y<<"), xur is "<<xur<<std::endl;
	    sn.x = xur; sn.ydir = 1;
	    it = br->tree.upper_bound(sn);
	    if(it == br->tree.end()) {
		ybl = br->miny;
	    } else {
		ybl = it->y;
	    }
	    if(br->tree.begin() == br->tree.end()) ybl = br->miny;
	    //std::cout<<"got point ("<<it->x<<","<<it->y<<"), ybl is "<<ybl<<std::endl;
	    if(isActive(xbl,ybl)) {
		//the region is valid, compute area
		int area = abs(yur-ybl)*abs(xur-xbl);
		//std::cout<<"active: "<<xbl<<","<<ybl<<":"<<xur<<","<<yur<<" = "<<area<<std::endl;
		if(area > max_area) {
		    //std::cout<<"15: "<<area<<std::endl;
		    max_area=area; xblf = xbl; yblf = ybl; xurf=xur; yurf=yur;
		    //std::cout<<"CHOOSE: ("<<xbl<<","<<ybl<<") : ("<<xur<<","<<yur<<")\n";
		}
	    } else {
		//std::cout<<"NOT active: "<<xbl<<","<<ybl<<":"<<xur<<","<<yur<<std::endl;
	    }

	    //case 16: keep ybl and look for the rest
	    //xur is in BR
	    sn.y = ybl; sn.xdir = -1; sn.ydir = 1;
	    it = br->tree.upper_bound(sn, &SegmentYCmp);
	    if(it == br->tree.begin()) {
		xur = br->maxx;
	    } else {
		it--;
		xur = it->x;
	    }
	    //std::cout<<"got point ("<<it->x<<","<<it->y<<"), xur is "<<xur<<std::endl;
	    sn.x = xur; sn.ydir = -1;
	    it = ur->tree.upper_bound(sn);
	    if(it == ur->tree.end()) {
		yur = ur->maxy;
	    } else {
		yur = it->y;
	    }
	    //std::cout<<"got point ("<<it->x<<","<<it->y<<"), yur is "<<yur<<std::endl;
	    sn.y = yur; sn.xdir = 1;
	    it = ul->tree.upper_bound(sn, &SegmentYCmp);
	    if(it == ul->tree.begin()) {
		xbl = ul->minx;
		//xbl = it->x;
	    } else {
		it --;
		xbl = it->x;
	    }
	    //std::cout<<"got point ("<<it->x<<","<<it->y<<"), xbl is "<<xbl<<std::endl;
	    if(isActive(xbl,ybl)) {
		//the region is valid, compute area
		int area = abs(yur-ybl)*abs(xur-xbl);
		//std::cout<<"active: "<<xbl<<","<<ybl<<":"<<xur<<","<<yur<<" = "<<area<<std::endl;
		if(area > max_area) {
		    //std::cout<<"16: "<<area<<std::endl;
		    max_area=area; xblf = xbl; yblf = ybl; xurf=xur; yurf=yur;
		    //std::cout<<"CHOOSE: ("<<xbl<<","<<ybl<<") : ("<<xur<<","<<yur<<")\n";
		}
	    } else {
		//std::cout<<"NOT active: "<<xbl<<","<<ybl<<":"<<xur<<","<<yur<<std::endl;
	    }
	}
    }
    //cases 5-8
    {
	boost::intrusive::rbtree<SegmentNode>::iterator ulit = ul->tree.begin();
	boost::intrusive::rbtree<SegmentNode>::iterator ulit_prev = ul->tree.begin();
	bool check_empty = (ul->tree.begin() == ul->tree.end());
	//choose both from BL
	while (ulit_prev != ul->tree.end() || check_empty) {
	    if(!check_empty) {
		if(ulit!=ul->tree.end()) {
		    yur = ulit->y;
		} else {
		    yur = ul->maxy;
		}
		if(ulit_prev == ulit) {
		    xbl = ul->minx;
		    ulit++;
		} else {
		    xbl = ulit_prev->x;
		    ulit++;
		    ulit_prev++;
		}
		if(!isActive(xbl,yur)) continue;
	    } else {
		check_empty = false;
		yur = ul->maxy;
		xbl = ul->minx;
	    }

	    //std::cout<<"\n\nchecking bottom corner at ("<<xbl<<",--) and (--,"<<yur<<")\n";

	    SegmentNode sn(xbl,yur,1,1);
	    //now we check for x upper bound in bl, the corresponding point's y is our ponential yur
	    boost::intrusive::rbtree<SegmentNode>::iterator blit = bl->tree.upper_bound(sn);
	    if(blit ==bl->tree.end()) {
		ybl = bl->miny;
		//std::cout<<"bl y at tree_end\n";
	    } else {
		ybl = blit->y;
	    }

	    //now we check for y upper bound in ur, the corresponding point's x is our ponential xur
	    sn.xdir = -1; sn.ydir = -1;
	    boost::intrusive::rbtree<SegmentNode>::iterator urit = ur->tree.upper_bound(sn, &SegmentYCmp);
	    if(urit == ur->tree.begin()) {
		xur = ur->maxx;
		//std::cout<<"ur x at tree_start\n";
	    } else {
		urit--;
		xur = urit->x;
	    }
	    //std::cout<<"check ("<<xur<<","<<ybl<<") in BR\n";
	    if(br->isActive(xur,ybl)) {
		//std::cout<<"active!\n";
		//the region is valid, compute area
		int area = abs(yur-ybl)*abs(xur-xbl);
		if(area > max_area) {
		    //std::cout<<"5-8: "<<area<<std::endl;
		    max_area=area; xblf = xbl; yblf = ybl; xurf=xur; yurf=yur;
		    //std::cout<<"CHOOSE: ("<<xbl<<","<<ybl<<") : ("<<xur<<","<<yur<<")\n";
		}
	    } else {
		//std::cout<<"not active, let's check all inside:\n";
		sn.x = xur; sn.y = ybl; sn.xdir=-1; sn.ydir=1;
		//region is actually bounded by ur corner, check the possible supports
		boost::intrusive::rbtree<SegmentNode>::iterator xur_it = br->tree.upper_bound(sn);
		boost::intrusive::rbtree<SegmentNode>::iterator yur_it = br->tree.upper_bound(sn, &SegmentYCmp);
		//find which one is first accoridng to x order
		//std::cout<<"check between "<<xur_it->x<<","<<xur_it->y<<" and "<<yur_it->x<<","<<yur_it->y<<"\n"; 
		if(xur_it->x*xur_it->xdir > yur_it->x*yur_it->xdir && yur_it != br->tree.end()) {
		    //std::cout<<"switchem\n";
		    //switch them
		    boost::intrusive::rbtree<SegmentNode>::iterator tmp = xur_it;
		    xur_it = yur_it;
		    yur_it = tmp;
		}
		boost::intrusive::rbtree<SegmentNode>::iterator tmp = xur_it;
		boost::intrusive::rbtree<SegmentNode>::iterator tmp2;
		while(tmp!=yur_it) {
		    tmp2=tmp;
		    tmp++;
		    int txur=tmp2->x;
		    int tybl= tmp!= br->tree.end()? tmp->y : br->miny;
		    txur = xur*br->xdir > txur*br->xdir ? xur : txur; 
		    tybl = ybl*br->ydir > tybl*br->ydir ? ybl : tybl; 
		    //std::cout<<"check ("<<txur<<","<<tybl<<")\n";
		    //double check to be on the safe side
		    if(!isActive(txur,tybl)) { 
			//std::cerr<<"messed up at ("<<txur<<","<<tybl<<")\n";
			continue;
		    }
		    if(!isActive(txur,yur)) { 
			//std::cerr<<"messed up at ("<<txur<<","<<yur<<")\n";
			continue;
		    }
		    if(!isActive(xbl,tybl)) { 
			//std::cerr<<"messed up at ("<<xbl<<","<<tybl<<")\n";
			continue;
		    }
		    //the region is valid, compute area
		    int area = abs(tybl-yur)*abs(txur-xbl);
		    if(area > max_area) {
			//std::cout<<"5-8(2): "<<area<<std::endl;
			max_area=area; xblf = xbl; yblf = tybl; xurf=txur; yurf=yur;
			//std::cout<<"CHOOSE: ("<<xbl<<","<<ybl<<") : ("<<xur<<","<<yur<<")\n";
		    }
		}
	    }
	}
    }
    //cases 9-11
    //std::cout<<"MORE CASESSSSSSSS\n\n";
    {
	boost::intrusive::rbtree<SegmentNode>::iterator urit = ur->tree.begin();
	boost::intrusive::rbtree<SegmentNode>::iterator urit_prev = ur->tree.begin();
	bool check_empty = (ur->tree.begin() == ur->tree.end());
	//choose both from BL
	while (urit_prev != ur->tree.end() || check_empty) {
	    if(!check_empty) {
		if(urit!=ur->tree.end()) {
		    yur = urit->y;
		} else {
		    yur = ur->maxy;
		}
		if(urit_prev == urit) {
		    xur = ur->maxx;
		    urit++;
		} else {
		    xur = urit_prev->x;
		    urit++;
		    urit_prev++;
		}
		if(!isActive(xur,yur)) continue;
	    } else {
		check_empty = false;
		xur = ur->maxx;
		yur = ur->maxy;
	    }
	    //std::cout<<"\n\nchecking bottom corner at ("<<xur<<","<<yur<<")\n";
	    SegmentNode sn(xur,yur,-1,1);
	    //now we check for x upper bound in ul, the corresponding point's y is our ponential yur
	    boost::intrusive::rbtree<SegmentNode>::iterator brit = br->tree.upper_bound(sn);
	    if(brit ==br->tree.end()) {
		ybl = br->miny;
		//std::cout<<"br y at tree_end\n";
	    } else {
		ybl = brit->y;
	    }
	    //now we check for y upper bound in br, the corresponding point's x is our ponential xur
	    sn.xdir = 1; sn.ydir = -1;
	    boost::intrusive::rbtree<SegmentNode>::iterator ulit = ul->tree.upper_bound(sn, &SegmentYCmp);
	    if(ulit == ul->tree.begin()) {
		xbl = ul->minx;
		//std::cout<<"ul x at tree_start\n";
	    } 
	    else //TST
		if(ulit ==ul->tree.end()) {
		    ulit--;
		    xbl = ulit->x;
		    //std::cout<<"ul x at tree_end\n";
		} else {
		    xbl = ulit->x;
		}
	    //std::cout<<"check ("<<xbl<<","<<ybl<<")\n";
	    if(bl->isActive(xbl,ybl)) {
		//std::cout<<"active!\n";
		//the region is valid, compute area
		int area = abs(yur-ybl)*abs(xur-xbl);
		if(area > max_area) {
		    //std::cout<<"9-12: "<<area<<std::endl;
		    max_area=area; xblf = xbl; yblf = ybl; xurf=xur; yurf=yur;
		    //std::cout<<"CHOOSE: ("<<xbl<<","<<ybl<<") : ("<<xur<<","<<yur<<")\n";
		}
	    } else {
		//std::cout<<"not active, let's check all inside:\n";
		sn.x = xbl; sn.y = ybl; sn.xdir=1; sn.ydir=1;
		//region is actually bounded by ur corner, check the possible supports
		boost::intrusive::rbtree<SegmentNode>::iterator xbl_it = bl->tree.upper_bound(sn);
		boost::intrusive::rbtree<SegmentNode>::iterator ybl_it = bl->tree.upper_bound(sn, &SegmentYCmp);
		//find which one is first accoridng to x order 
		if(xbl_it->x*xbl_it->xdir > ybl_it->x*ybl_it->xdir && ybl_it != bl->tree.end()) {
		    //switch them
		    boost::intrusive::rbtree<SegmentNode>::iterator tmp = xbl_it;
		    xbl_it = ybl_it;
		    ybl_it = tmp;
		}
		boost::intrusive::rbtree<SegmentNode>::iterator tmp = xbl_it;
		boost::intrusive::rbtree<SegmentNode>::iterator tmp2;
		while(tmp!=ybl_it) {
		    tmp2=tmp;
		    tmp++;
		    int txbl=tmp2->x;
		    int tybl=tmp!=bl->tree.end() ? tmp->y : bl->miny;
		    txbl = xbl*bl->xdir > txbl*bl->xdir ? xbl : txbl; 
		    tybl = ybl*bl->ydir > tybl*bl->ydir ? ybl : tybl; 
		    //std::cout<<"check ("<<txbl<<","<<tybl<<")\n";
		    //double check to be on the safe side
		    if(!isActive(txbl,tybl)) { 
			//std::cerr<<"messed up at ("<<txbl<<","<<tybl<<")\n";
			continue;
		    }
		    if(!isActive(xur,tybl)) { 
			//std::cerr<<"messed up at ("<<xur<<","<<tybl<<")\n";
			continue;
		    }
		    if(!isActive(txbl,yur)) { 
			//std::cerr<<"messed up at ("<<txbl<<","<<yur<<")\n";
			continue;
		    }
		    //the region is valid, compute area
		    int area = abs(yur-tybl)*abs(xur-txbl);
		    if(area > max_area) {
			//std::cout<<"9-12(2): "<<area<<std::endl;
			max_area=area; xblf = txbl; yblf = tybl; xurf=xur; yurf=yur;
			//std::cout<<"CHOOSE: ("<<xbl<<","<<ybl<<") : ("<<xur<<","<<yur<<")\n";
		    }
		}
	    }
	}
    }
    //cases 12-14
    //std::cout<<"EVEN MORE CASESSSSSSSSSSSS\n\n";
    {
	boost::intrusive::rbtree<SegmentNode>::iterator brit = br->tree.begin();
	boost::intrusive::rbtree<SegmentNode>::iterator brit_prev = br->tree.begin();
	bool check_empty = (br->tree.begin() == br->tree.end());
	//choose both from BL
	while (brit_prev != br->tree.end() || check_empty) {
	    if(!check_empty) {
		if(brit!=br->tree.end()) {
		    ybl = brit->y;
		} else {
		    ybl = br->miny;
		}
		if(brit_prev == brit) {
		    xur = br->maxx;
		    brit++;
		} else {
		    xur = brit_prev->x;
		    brit++;
		    brit_prev++;
		}
		if(!isActive(xur,ybl)) continue;
	    } else {
		check_empty = false;
		xur = br->maxx;
		ybl = br->miny;
	    }
	    //std::cout<<"\n\nchecking bottom corner at ("<<xur<<", --),(--,"<<ybl<<")\n";
	    SegmentNode sn(xur,ybl,-1,-1);
	    //now we check for x upper bound in ul, the corresponding point's y is our ponential yur
	    boost::intrusive::rbtree<SegmentNode>::iterator urit = ur->tree.upper_bound(sn);
	    if(urit ==ur->tree.end()) {
		yur = ur->maxy;
		//std::cout<<"ur y at tree_end\n";
	    } else {
		yur = urit->y;
	    }
	    //now we check for y upper bound in br, the corresponding point's x is our ponential xur
	    sn.xdir = 1; sn.ydir = 1;
	    boost::intrusive::rbtree<SegmentNode>::iterator blit = bl->tree.upper_bound(sn, &SegmentYCmp);
	    //TST
	    if(blit == bl->tree.begin()) {
		xbl = bl->minx;
		//std::cout<<"bl x at tree_start\n";
	    } 
	    else 
		if(blit ==bl->tree.end()) {
		    blit--;
		    xbl = blit->x;
		    //std::cout<<"bl x at tree_end\n";
		} else {
		    xbl = blit->x;
		}
	    //std::cout<<"check ("<<xbl<<","<<yur<<")\n";
	    if(ul->isActive(xbl,yur)) {
		//std::cout<<"active!\n";
		//the region is valid, compute area
		int area = abs(yur-ybl)*abs(xur-xbl);
		if(area > max_area) {
		    //std::cout<<"13-15: "<<area<<std::endl;
		    max_area=area; xblf = xbl; yblf = ybl; xurf=xur; yurf=yur;
		    //std::cout<<"CHOOSE: ("<<xbl<<","<<ybl<<") : ("<<xur<<","<<yur<<")\n";
		}
	    } else {
		//std::cout<<"not active, let's check all inside:\n";
		sn.x = xbl; sn.y = yur; sn.xdir=1; sn.ydir=-1;
		//region is actually bounded by ur corner, check the possible supports
		boost::intrusive::rbtree<SegmentNode>::iterator xul_it = ul->tree.upper_bound(sn);
		boost::intrusive::rbtree<SegmentNode>::iterator yul_it = ul->tree.upper_bound(sn, &SegmentYCmp);
		//find which one is first accoridng to x order 
		if(xul_it->x*xul_it->xdir > yul_it->x*yul_it->xdir && yul_it != ul->tree.end()) {
		    //switch them
		    //std::cout<<"switchem\n";
		    boost::intrusive::rbtree<SegmentNode>::iterator tmp = xul_it;
		    xul_it = yul_it;
		    yul_it = tmp;
		}
		boost::intrusive::rbtree<SegmentNode>::iterator tmp = xul_it;
		boost::intrusive::rbtree<SegmentNode>::iterator tmp2;
		while(tmp!=yul_it) {
		    tmp2=tmp;
		    tmp++;
		    int txbl=tmp2->x;
		    int tyur= tmp!=ul->tree.end() ? tmp->y : ul->maxy;
		    //std::cout<<"original : "<<txbl<<","<<tyur<<std::endl;
		    txbl = xbl*ul->xdir > txbl*ul->xdir ? xbl : txbl; 
		    tyur = yur*ul->ydir > tyur*ul->ydir ? yur : tyur; 
		    //std::cout<<"check ("<<txbl<<","<<tyur<<")\n";
		    //double check to be on the safe side
		    if(!isActive(txbl,tyur)) { 
			//std::cerr<<"messed up at ("<<txbl<<","<<tyur<<")\n";
			continue;
		    }
		    if(!isActive(txbl,ybl)) { 
			//std::cerr<<"messed up at ("<<txbl<<","<<ybl<<")\n";
			continue;
		    }
		    if(!isActive(xur,tyur)) { 
			//std::cerr<<"messed up at ("<<xur<<","<<tyur<<")\n";
			continue;
		    }
		    //the region is valid, compute area
		    int area = abs(tyur-ybl)*abs(xur-txbl);
		    if(area > max_area) {
			//std::cout<<"13-15(2): "<<area<<std::endl;
			max_area=area; xblf = txbl; yblf = ybl; xurf=xur; yurf=tyur;
			//std::cout<<"CHOOSE: ("<<xbl<<","<<ybl<<") : ("<<xur<<","<<yur<<")\n";
		    }
		}
	    }
	}
    }
    //case 15
    //case 16
    if(!isActive(xblf,yurf)) { 
	//std::cout<<"xblyur messed up at ("<<xblf<<","<<yurf<<")\n";
	return -1;
    }
    if(!isActive(xurf,yblf)) { 
	//std::cout<<"xurybl messed up at ("<<xurf<<","<<yblf<<")\n";
	return -1;
    }
    if(!isActive(xblf,yblf)) { 
	//std::cout<<"xblybl messed up at ("<<xblf<<","<<yblf<<")\n";
	return -1;
    }
    if(!isActive(xurf,yurf)) { 
	//std::cout<<"xuryur messed up at ("<<xurf<<","<<yurf<<")\n";
	return -1;
    }

    return max_area;
}
	
CellIdxCube MaxEmptyCubeExtractor::getMaxCube(SimpleOccMap *map) {
    CellIndex id;
    CellIdxCube cube;
    for(id.k = map->size_z-1; id.k >0 ; --id.k) {
	//sweep plane is along z
	//TODO: handle top and bottom
	for(id.i = 0; id.i < map->size_x; ++id.i) {
	    for(id.j = 0; id.j < map->size_y; ++id.j) {
		//std::cout<<id.i<<","<<id.j<<","<<id.k<<std::endl;
		if(map->grid[id.i][id.j][id.k] > SimpleOccMap::FREE ) {
		    //std::cout<<"found occupied\n";
		    //cell is occupied, go through active sets
		    for(int i=0; i<visited.size(); ++i) {
			if(visited[i]->z == id.k) break;
			if(visited[i]->isActive(id.i,id.j)) {
			    if(visited[i]->maxRectangle(cube.bl.i,cube.bl.j,cube.ur.i,cube.ur.j) > 0) {
				//TST
				cube.bl.i++; cube.bl.j++;
				cube.ur.i--; cube.ur.j--;
				cube.bl.k = id.k+1;
				cube.ur.k = visited[i]->z-1;
				empty_cubes.push_back(cube);
			    }
			    visited[i]->insertPoint(id.i,id.j); 
		    	}
		    }
		    ActiveSet *as = new ActiveSet(id.i,id.j,id.k,0,map->size_x-1,0,map->size_y-1);
		    visited.push_back(as);
		}
	    }
	}    
    }

    sort(empty_cubes.begin(), empty_cubes.end(), cubecmpr);
    for(int i=0; i<visited.size(); ++i) {
	delete visited[i];
    }
    visited.clear();
    return empty_cubes.front();

}

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
				    minval(distance_grid[id.i][id.j][id.k][0],distance_grid[id.i+1][id.j][id.k][0]+1) : minval(distance_grid[id.i][id.j][id.k][0],borderI), map->size_x);
				distance_grid[id.i][id.j][id.k][1] = minval(id.j < map->size_y-1 ?
				    minval(distance_grid[id.i][id.j][id.k][1],distance_grid[id.i][id.j+1][id.k][1]+1) : minval(distance_grid[id.i][id.j][id.k][1],borderJ), map->size_y);
				distance_grid[id.i][id.j][id.k][2] = minval(id.k < map->size_z-1 ?
				    minval(distance_grid[id.i][id.j][id.k][2],distance_grid[id.i][id.j][id.k+1][2]+1) : minval(distance_grid[id.i][id.j][id.k][2],borderK), map->size_z);
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
    for(id.i = 0; id.i < map->size_x; ++id.i) {
	    for(id.j = 0; id.j < map->size_y; ++id.j) {
		    for(id.k = 0; id.k < map->size_z; ++id.k) {
			    Triplet thisone = distance_grid.at(id);
			    //thisone[0]  = distance_grid[id.i][id.j][id.k][0];
			    //thisone[1]  = distance_grid[id.i][id.j][id.k][1];
			    //thisone[2]  = distance_grid[id.i][id.j][id.k][2];
			    if((thisone[0])*(thisone[1])*(thisone[2]) > maxvolume) {
				//check what the maxvolume at this point would be
				Triplet min_ids[6];
				for(int x=0; x<6; ++x) {
				    min_ids[x] = thisone;
					//min_ids[x][0] = thisone[0];
					//min_ids[x][1] = thisone[1];
					//min_ids[x][2] = thisone[2];
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
//					std::cerr<<"NEW "<<maxvolu_id<<" "<<maxvolume<<" "<<id.i<<" "<<id.j<<" "<<id.k<<": is :"<<min_ids[maxvolu_id][0]<<" "<<min_ids[maxvolu_id][1]<<" "<<min_ids[maxvolu_id][2]<<" was: "<<thisone[0]<<" "<<thisone[1]<<" "<<thisone[2]<<std::endl;
//					std::cerr<<"CUBE ("<<cube.bl.i<<","<<cube.bl.j<<","<<cube.bl.k<<") : ("<<cube.ur.i<<","<<cube.ur.j<<","<<cube.ur.k<<") volume "<<cube.volume()<<std::endl;
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
