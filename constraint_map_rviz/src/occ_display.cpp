#include <OGRE/OgreSceneNode.h>
#include <OGRE/OgreSceneManager.h>

#include <tf/transform_listener.h>

#include <rviz/visualization_manager.h>
#include <rviz/properties/color_property.h>
#include <rviz/properties/float_property.h>
#include <rviz/properties/int_property.h>
#include <rviz/frame_manager.h>

#include "occ_visual.hpp"
#include "occ_display.hpp"
#include <constraint_map/simple_occ_map.h>

OCCDisplay::OCCDisplay(){
    color_property_ = new rviz::ColorProperty( "Color", QColor( 200, 20, 20 ),
	    "Color to draw occupied voxels.",
	    this, SLOT( updateColorAndAlpha() ));

    alpha_property_ = new rviz::FloatProperty( "Alpha", 1.0,
	    "0 is fully transparent, 1.0 is fully opaque.",
	    this, SLOT( updateColorAndAlpha() ));
    
    free_color_property_ = new rviz::ColorProperty( "FreeColor", QColor( 50, 200, 50 ),
	    "Color to draw free voxels.",
	    this, SLOT( updateColorAndAlpha() ));

    free_alpha_property_ = new rviz::FloatProperty( "FreeAlpha", 0.2,
	    "0 is fully transparent, 1.0 is fully opaque.",
	    this, SLOT( updateColorAndAlpha() ));
    
    show_free_property_ = new rviz::BoolProperty( "Show Free Space", false,
	    "show free space or no",
	    this, SLOT( updateColorAndAlpha() ));
}

void OCCDisplay::onInitialize(){
    MFDClass::onInitialize();
}

OCCDisplay::~OCCDisplay(){
}
void OCCDisplay::reset(){
    MFDClass::reset();
    visuals_.clear();
    fvisuals_.clear();
}
void OCCDisplay::updateColorAndAlpha(){
    float alpha=alpha_property_->getFloat();
    Ogre::ColourValue color=color_property_->getOgreColor();
    
    for(size_t i=0;i<visuals_.size();i++){
	visuals_[i]->setColor(color.r,color.g,color.b,alpha);
    }
    
    if(show_free_property_->getBool()) {
	float falpha=free_alpha_property_->getFloat();
	Ogre::ColourValue fcolor=free_color_property_->getOgreColor();
	for(size_t i=0;i<fvisuals_.size();i++){
	    fvisuals_[i]->setColor(fcolor.r,fcolor.g,fcolor.b,falpha);
	}
    }
}

//FIXME
void OCCDisplay::processMessage( const constraint_map::SimpleOccMapMsg::ConstPtr& msg ){
    Ogre::Quaternion orientation;
    Ogre::Vector3 position;
    if( !context_->getFrameManager()->getTransform( msg->header.frame_id,msg->header.stamp,position, orientation)){
	ROS_DEBUG( "Error transforming from frame '%s' to frame '%s'",msg->header.frame_id.c_str(), qPrintable( fixed_frame_ ));
	return;
    }
    //TODO?
    visuals_.clear();
    fvisuals_.clear();

    SimpleOccMap *map = new SimpleOccMap();
    map->fromMessage(*msg);

    std::vector<CellIndex> occupied;
    map->getOccupied(occupied);
    Ogre::ColourValue color=color_property_->getOgreColor();
    float alpha = alpha_property_->getFloat();

    for(int itr=0;itr<occupied.size();itr++){
	boost::shared_ptr<OCCVisual> visual;

	visual.reset(new OCCVisual(context_->getSceneManager(), scene_node_));
	
	Eigen::Vector3f center_cell;
	map->getCenterCell(occupied[itr],center_cell);

	Ogre::Vector3 cell_pos(center_cell(0),center_cell(1),center_cell(2));
	
	visual->setCell(cell_pos,msg->cell_size);
	visual->setFramePosition(position);
	visual->setFrameOrientation(orientation);
	visual->setColor(color.r,color.g,color.b,alpha);
	visuals_.push_back(visual);
    }

    if(show_free_property_->getBool()) {
	Ogre::ColourValue fcolor=free_color_property_->getOgreColor();
	float falpha = free_alpha_property_->getFloat();

	std::vector<CellIndex> free;
	map->getFree(free);
    
	for(int itr=0;itr<free.size();itr++){
	    boost::shared_ptr<OCCVisual> visual;

	    visual.reset(new OCCVisual(context_->getSceneManager(), scene_node_));

	    Eigen::Vector3f center_cell;
	    map->getCenterCell(free[itr],center_cell);

	    Ogre::Vector3 cell_pos(center_cell(0),center_cell(1),center_cell(2));

	    visual->setCell(cell_pos,msg->cell_size);
	    visual->setFramePosition(position);
	    visual->setFrameOrientation(orientation);
	    visual->setColor(fcolor.r,fcolor.g,fcolor.b,falpha);
	    fvisuals_.push_back(visual);
	}

    }
    delete map;
}

#include <pluginlib/class_list_macros.h>
    PLUGINLIB_EXPORT_CLASS(OCCDisplay,rviz::Display)

