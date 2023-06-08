/*This class provides the basic operation that all gazebo plugins need to do with the startup function 
(linking the callback to the simulation, get joints number and names).
All the rest must be implemented in the specific plugin, which must inherite from this class*/

#include "BasicPlugin.h"

namespace gazebo
{

  void BasicPlugin::startup(physics::ModelPtr parent, sdf::ElementPtr _sdf)
  {
    int argc = 0;
    char **argv = NULL;
    if (!ros::isInitialized())
    {
      ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. "
                       << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
      return;
    }

    else
    {
      ROS_INFO("***** %s *****", plugin_name.c_str());

      // Store the pointer to the model
      this->model = parent;

      // count number of robot joints
      // The first joint is the fixed worl djoint
      joints_count = this->model->GetJointCount();
      // ROS_INFO("%s n_joints found: %d", plugin_name.c_str(), joints_count);

      // retrive joints information
      this->gazebo_joints = this->model->GetJoints();

      // Listen to the update event. This event is broadcast every
      // simulation iteration.
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(gazebo_callback);
      // get joints name
      // index starts from 1 because the first values corresponds to the fixed joint 'world', which is NaN
      for (int i = 1; i <= joints_count - 1; i++)
      {
        joints.name.push_back(gazebo_joints[i]->GetName());
        // ROS_INFO("%s joint_name: %s", plugin_name.c_str(), joints.name.back().c_str());
      }
    }
  }

}