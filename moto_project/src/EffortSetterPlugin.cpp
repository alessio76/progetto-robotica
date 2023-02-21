/*This plugin implements the physical interface to the robot, that is it uses the desidered joint effort to move the robot. 
This effort is read from the /gazebo/effort_cmd topic. */

#include "SetterPlugin.h"

namespace gazebo
{
  
  class EffortSetterPlugin : public SetterPlugin
  {

  private:
    /*When using a member function as a callback the standard procedure for subcribing is slightly more complicated:
      1) you need to change the signature of the callback function
      from void effortCallback(const sensor_msgs::JointState& joint_states)
      to void effortCallback(const boost::shared_ptr<sensor_msgs::JointState const> joint_states)

      2) you need to pass as a template argument the type of the message because the compiler can no longer deduce it
      so we change this nh.subscribe(effort_topic_name, 1000, callback);
      in nh.subscribe<sensor_msgs::JointState>(effort_topic_name, 1000, boost::bind(&EffortSetter::effortCallback, this,_1));

      or alternatively in boost::function<void(const boost::shared_ptr<sensor_msgs::JointState const>&)> callback =
                          boost::bind(&EffortSetter::effortCallback, this,_1);

      effort_sub = nh.subscribe<sensor_msgs::JointState>(effort_topic_name, 1000, callback);

      Although these constructs are much more versatile, for simplicity there is a built-in functions to construct
      the subscriber directly, that is
      effort_sub = nh.subscribe(effort_topic_name, 1000, &EffortSetter::effortCallback, this);
      which doesn't require changing the signature of the callbakc, that remains
      void effortCallback(const sensor_msgs::JointState& joint_states)*/

    // callback for copying published effort from topic to internal variables
    void effortCallback(const boost::shared_ptr<sensor_msgs::JointState const> joint_states)
    {
      joints.effort = joint_states->effort;
    }

     void onUpdate_action() override
    {
      // set joint effort
        for (int i = 1; i <= joints_count - 1; i++)
        {
          gazebo_joints[i]->SetForce(0,joints.effort[i]);
        };
    }

  public:
    EffortSetterPlugin()
    {
      desidered_topic_param="desidered_effort_topic_param";
      plugin_name = "gazebo_effort_setter_plugin";
      gazebo_callback = boost::bind(&EffortSetterPlugin::OnUpdate, this);
      subscriber_callback = boost::bind(&EffortSetterPlugin::effortCallback, this, _1);
    }

    void Load(physics::ModelPtr parent, sdf::ElementPtr _sdf)
    {

      this->startup(parent, _sdf);
      if(this->subscribe_topic()) ROS_INFO("Effort subscriber active");
      joints.effort.resize(joints_count);

    }

    // Called by the world update start event
    void OnUpdate()
    {
      // clear previous values
      joints.effort.clear();
      this->onUpdate_function();
    }
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(EffortSetterPlugin)
}