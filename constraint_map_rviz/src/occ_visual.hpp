#ifndef NDT_VISUAL_H
#define NDT_VISUAL_H

namespace Ogre{
  class Vector3;
  class Quaternion;
}

namespace rviz{
  class Shape;
}

class OCCVisual{
    public:
	OCCVisual( Ogre::SceneManager* scene_manager, Ogre::SceneNode* parent_node );
	virtual ~OCCVisual();
	void setCell(Ogre::Vector3 position, double resolution);
	void setFramePosition(const Ogre::Vector3& position);
	void setFrameOrientation(const Ogre::Quaternion& orientation);
	void setColor( float r, float g, float b, float a );
    private:
	boost::shared_ptr<rviz::Shape> voxel_;
	Ogre::SceneNode* frame_node_;
	Ogre::SceneManager* scene_manager_;
};
#endif 
