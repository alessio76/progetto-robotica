#pragma once

#include <functional>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ros/ros.h>
#include "sensor_msgs/JointState.h"

namespace gazebo
{
  class BasicPlugin : public ModelPlugin
  {
    // Pointer to the model
  protected:
    physics::ModelPtr model;
    // Pointer to the update event connection
    event::ConnectionPtr updateConnection;
    // callback to be linked with the gazebo simulation, that is the OnUpdate function of the specific class
    boost::function<void(const gazebo::common::UpdateInfo &)> gazebo_callback;

    // gazebo vector
    physics::Joint_V gazebo_joints;
    int joints_count;
    ros::NodeHandle nh;
    std::string plugin_name;
    // joints
    sensor_msgs::JointState joints;
    void startup(physics::ModelPtr parent, sdf::ElementPtr _sdf);

  public:
    virtual void Load(physics::ModelPtr parent, sdf::ElementPtr _sdf) = 0;
    virtual void OnUpdate() = 0;
  };
}
