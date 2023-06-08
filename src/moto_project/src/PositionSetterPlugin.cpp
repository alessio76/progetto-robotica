/*This plugin implements the pure kinematic movement of the joints by directly setting the joints' position. 
Tha desidered joint positions are read from the /gazebo/position_cmd topic*/

#include "SetterPlugin.h"

namespace gazebo
{

  class PositionSetterPlugin : public SetterPlugin
  {

  private:
    // callback for copying published positions from topic to internal variables
   void positionCallback(const boost::shared_ptr<sensor_msgs::JointState const> joint_states)
    {
      joints.position = joint_states->position;
    }

     void onUpdate_action() override
    {
      // set joint position
        for (int i = 1; i <= joints_count - 1; i++)
        {
          gazebo_joints[i]->SetPosition(0,joints.position[i]);
        };
    }

  public:
  PositionSetterPlugin(){
      desidered_topic_param="desidered_position_topic_param";
      plugin_name = "gazebo_position_setter_plugin";
      gazebo_callback = boost::bind(&PositionSetterPlugin::OnUpdate, this);
      subscriber_callback = boost::bind(&PositionSetterPlugin::positionCallback, this, _1);


  }
     void Load(physics::ModelPtr parent, sdf::ElementPtr _sdf)
    {

      this->startup(parent, _sdf);
      if(this->subscribe_topic()) ROS_INFO("Position subscriber active");
      joints.position.resize(joints_count);

    }

    // Called by the world update start event
    void OnUpdate()
    {
      // clear previous values
      joints.position.clear();
      this->onUpdate_function();
    }
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(PositionSetterPlugin)
}