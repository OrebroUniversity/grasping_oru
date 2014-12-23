#include <OGRE/OgreVector3.h>
#include <OGRE/OgreSceneNode.h>
#include <OGRE/OgreSceneManager.h>
#include <Eigen/Dense>
#include <ros/ros.h>
#include <rviz/ogre_helpers/shape.h>

#include "occ_visual.hpp"

OCCVisual::OCCVisual( Ogre::SceneManager* scene_manager, Ogre::SceneNode* parent_node ){
    scene_manager_ = scene_manager;
    frame_node_ = parent_node->createChildSceneNode();
    voxel_.reset(new rviz::Shape(rviz::Shape::Cube,scene_manager_,frame_node_ ));
}

OCCVisual::~OCCVisual()
{
    scene_manager_->destroySceneNode( frame_node_ );
}

void OCCVisual::setCell(Ogre::Vector3 position, double resolution){
    Ogre::Vector3 scale(resolution,resolution,resolution);
    voxel_->setScale(scale);
    voxel_->setPosition(position);
}

void OCCVisual::setFramePosition( const Ogre::Vector3& position ){
    frame_node_->setPosition( position );
}

void OCCVisual::setFrameOrientation( const Ogre::Quaternion& orientation ){
    frame_node_->setOrientation( orientation );
}

void OCCVisual::setColor( float r, float g, float b, float a ){
    voxel_->setColor( r, g, b, a );
}

