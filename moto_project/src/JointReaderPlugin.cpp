/*This plugin publishes the joint variables on the topic /gazebo/get_joints, to which joint_states subscribes to and publishes the joint variables for the ros network */

#include "BasicPlugin.h"

namespace gazebo
{
  class JointReaderPlugin : public BasicPlugin
  {

  private:
    ros::Publisher joint_pub;
    const std::string topic_name{"/gazebo/get_joints"};

  public:
    JointReaderPlugin() 
    {
      plugin_name = "gazebo_joint_reader";
      gazebo_callback = std::bind(&JointReaderPlugin::OnUpdate, this);
    }

    void Load(physics::ModelPtr parent, sdf::ElementPtr _sdf)
    {
      this->startup(parent, _sdf);

      // create the topic
      this->joint_pub = nh.advertise<sensor_msgs::JointState>(topic_name, 1000);
    }

    // Called by the world update start event
  public:
    void OnUpdate()
    {
      if (ros::ok())
      {
        // retrive joints information
        this->gazebo_joints = this->model->GetJoints();

        // clear previous values
        joints.position.clear();
        joints.velocity.clear();

        // get joint variables
        for (int i = 1; i <= joints_count - 1; i++)
        {
          joints.position.push_back(gazebo_joints[i]->Position());
          joints.velocity.push_back(gazebo_joints[i]->GetVelocity(0));
        }

        // set current time
        joints.header.stamp = ros::Time::now();

        // publish joint variables
        joint_pub.publish(joints);
        ros::spinOnce();
      }
    }
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(JointReaderPlugin)
}
