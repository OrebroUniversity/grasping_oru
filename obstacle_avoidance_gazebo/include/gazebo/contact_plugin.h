#pragma once
#include <string>

#include <gazebo/gazebo.hh>
#include <gazebo/sensors/sensors.hh>
#include <gazebo_msgs/ContactsState.h>

#include <ros/ros.h>

namespace gazebo {
/// \brief An example plugin for a contact sensor.
class ContactPlugin : public SensorPlugin {
 public:
  /// \brief Constructor.
  ContactPlugin();

  /// \brief Destructor.
  virtual ~ContactPlugin();

  /// \brief Load the sensor plugin.
  /// \param[in] _sensor Pointer to the sensor that loaded this plugin.
  /// \param[in] _sdf SDF element that describes the plugin.
  virtual void Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf);

 private:
  /// \brief A nodehandle to the ros node.
  ros::NodeHandle nH;

  /// \brief A publisher to ros.
  ros::Publisher contactsStatePub;
  
  /// \brief Callback that receives the contact sensor's update signal.
  virtual void OnUpdate();

  /// \brief Pointer to the contact sensor
  sensors::ContactSensorPtr parentSensor;

  /// \brief Connection that maintains a link between the contact sensor's
  /// updated signal and the OnUpdate callback.
  event::ConnectionPtr updateConnection;
};
}
