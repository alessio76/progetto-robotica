/*This class contains the basic function that all setter plugin need to do. 
The base scheme is that the generic setter plugin reads the informations thanks to the function offered by
BasicPlugin and then create a subscriber in order to read from a ros topic the values to send to the simulation*/

#pragma once
#include "SetterPlugin.h"

namespace gazebo
{
    bool SetterPlugin::subscribe_topic()
    {
        // first check that the name parameter exists
        if (nh.hasParam(desidered_topic_param))
        {
            // take that parameter
            nh.getParam(desidered_topic_param, desidered_topic_name);
            // subscribe to topic
            subscriber = nh.subscribe<sensor_msgs::JointState>(desidered_topic_name, 1000, subscriber_callback);
            return true;
        }

        else
        {
            ROS_ERROR("Parameter not found");
            return false;
        }
    }

    void SetterPlugin::onUpdate_function()
    {
        if (ros::ok())
        {
            // retrive joints
            this->gazebo_joints = this->model->GetJoints();

            //depeding on the specific setter this function actually implements the action to be performed
            onUpdate_action();

            ros::spinOnce();
        }
    }
}