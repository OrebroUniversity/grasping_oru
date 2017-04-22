#include <gazebo/contact_plugin.h>

GZ_REGISTER_SENSOR_PLUGIN(gazebo::ContactPlugin)

namespace gazebo {
/////////////////////////////////////////////////
ContactPlugin::ContactPlugin() : SensorPlugin() {
  contactsStatePub = nH.advertise<gazebo_msgs::ContactsState>("contacts", 1);
}

/////////////////////////////////////////////////
ContactPlugin::~ContactPlugin() {}

/////////////////////////////////////////////////
void ContactPlugin::Load(sensors::SensorPtr _sensor, sdf::ElementPtr /*_sdf*/) {
  // Get the parent sensor.
  this->parentSensor =
      std::dynamic_pointer_cast<sensors::ContactSensor>(_sensor);

  // Make sure the parent sensor is valid.
  if (!this->parentSensor) {
    gzerr << "gazebo::ContactPlugin requires a ContactSensor.\n";
    return;
  }

  // Connect to the sensor update event.
  this->updateConnection = this->parentSensor->ConnectUpdated(
      std::bind(&ContactPlugin::OnUpdate, this));

  // Make sure the parent sensor is active.
  this->parentSensor->SetActive(true);
}

/////////////////////////////////////////////////
void ContactPlugin::OnUpdate() {
  gazebo_msgs::ContactsState contactsStateMsg;
  contactsStateMsg.header.stamp = ros::Time::now();
  // Get all the contacts.
  msgs::Contacts contacts;
  contacts = this->parentSensor->Contacts();
  for (unsigned int i = 0; i < contacts.contact_size(); ++i) {
    
    gazebo_msgs::ContactState contactStateMsg;
    contactStateMsg.collision1_name = contacts.contact(i).collision1();
    contactStateMsg.collision2_name = contacts.contact(i).collision2();
    
    for (unsigned int j = 0; j < contacts.contact(i).position_size(); ++j) {
      geometry_msgs::Vector3 contact_position;
      geometry_msgs::Vector3 contact_normal;

      contact_position.x = contacts.contact(i).position(j).x();
      contact_position.y = contacts.contact(i).position(j).y();
      contact_position.z = contacts.contact(i).position(j).z();

      contact_normal.x = contacts.contact(i).normal(j).x();
      contact_normal.y = contacts.contact(i).normal(j).y();
      contact_normal.z = contacts.contact(i).normal(j).z();
      
      contactStateMsg.contact_positions.push_back(contact_position);
      contactStateMsg.contact_normals.push_back(contact_normal);
      contactStateMsg.depths.push_back(contacts.contact(i).depth(j));
    }
    contactsStateMsg.states.push_back(contactStateMsg);
  }

  contactsStatePub.publish(contactsStateMsg);
}
}
