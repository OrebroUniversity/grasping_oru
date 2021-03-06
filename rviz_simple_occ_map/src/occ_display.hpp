#ifndef NDT_DISPLAY_H
#define NDT_DISPLAY_H

#ifndef Q_MOC_RUN
#include <boost/circular_buffer.hpp>
#include <constraint_map/SimpleOccMapMsg.h>
#include <rviz/message_filter_display.h>
#endif

namespace Ogre
{
  class SceneNode;
}

namespace rviz
{
  class ColorProperty;
  class FloatProperty;
  class IntProperty;
}


class OCCVisual;

class OCCDisplay: public rviz::MessageFilterDisplay<constraint_map::SimpleOccMapMsg>{
    Q_OBJECT
    public:

	OCCDisplay();
	virtual ~OCCDisplay();

    protected:
	virtual void onInitialize();

	virtual void reset();

	private Q_SLOTS:
	    void updateColorAndAlpha();
	    void updateHistoryLength();

    private:
	void processMessage(const constraint_map::SimpleOccMapMsg::ConstPtr& msg);

	std::vector<boost::shared_ptr<OCCVisual> > visuals_;
	std::vector<boost::shared_ptr<OCCVisual> > fvisuals_;

	rviz::ColorProperty* color_property_;
	rviz::FloatProperty* alpha_property_;
	
	rviz::BoolProperty* show_free_property_;
	rviz::ColorProperty* free_color_property_;
	rviz::FloatProperty* free_alpha_property_;
	//rviz::IntProperty* history_length_property_;
};

#endif 

