#pragma once

#include "BasicPlugin.h"

namespace gazebo
{

  class SetterPlugin : public BasicPlugin
  {

  protected:
    ros::Subscriber subscriber;
    // function that the subscriber needs to execute
    boost::function<void(const boost::shared_ptr<sensor_msgs::JointState const> &)> subscriber_callback;

    // parameter on the parameter server which stores the name of the topic that publishes
    std::string desidered_topic_param;

    // actual name of the topic, taken from the parameter server
    std::string desidered_topic_name;

    //method that read the parameter and create the subscriber that reads desidered values
    bool subscribe_topic();

    //specif method each particular setter needs to implement to actually execute something
    virtual void onUpdate_action() = 0;

    //wrapper method that decouples the OnUpdate that needs to be overridden from the actual action of the plugin
    void onUpdate_function();

    public:
    virtual void Load(physics::ModelPtr parent, sdf::ElementPtr _sdf) = 0;
    virtual void OnUpdate() = 0;
  };
}