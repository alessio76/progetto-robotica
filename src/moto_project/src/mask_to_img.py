#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from moto_project.msg import mask
from std_msgs.msg import Int8MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class Mask_to_image:
	def __init__(self,mask_source_topic,mask_image_topic):
		self.image_pub = rospy.Publisher(mask_image_topic, Image, queue_size=10)
		self.mask_source=rospy.Subscriber( mask_source_topic, mask, self.callback)
		self.bridge = CvBridge()
	
	def callback(self,data):
	# Reshape the received data into a 2D numpy array
	    image_data = np.array(data.mask.data, dtype=np.uint8)
	    image_data = image_data.reshape((data.mask.layout.dim[0].size, data.mask.layout.dim[1].size))
	    # Convert the numpy array to an OpenCV image
	    cv_image = image_data*255
	    
	    # Convert the OpenCV image to a ROS image message
	    ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="mono8")
	    ros_image.header.stamp=data.header.stamp
	    ros_image.header.seq=data.header.seq
	 
            
	    
	    # Publish the ROS image message
	    self.image_pub.publish(ros_image)

if __name__ == '__main__':
    rospy.init_node('mask_to_image_converter', anonymous=True)
    mask_image_topic=rospy.get_param("mask_image_topic_name")
    # Create a subscriber for the Int8MultiArray topic
    mask_topic=rospy.get_param("mask_topic_name")
    mask_processor=Mask_to_image(mask_topic,mask_image_topic)
    rospy.spin()
