#!/usr/bin/env python3

import os
from moto_project.msg import mask
import random
import copy
import numpy as np
from PIL import Image
import cv2
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import message_filters
import warnings
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import quaternion_matrix, quaternion_from_matrix


warnings.filterwarnings("ignore")


xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])

def get_bbox(label):
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    img_width = 480
    img_length = 640
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

class ImageProcessor:
    def __init__(self,image_topic,depth_topic,mask_topic,pose_model,pose_refinement_model,class_id_dict,num_points,minimun_num_pt,cam_scale=1):
        self.bridge = CvBridge()
        self.depth_topic=depth_topic
        self.cam_scale=cam_scale
        self.image_topic=image_topic
        self.num_points=num_points
        self.tf_broadcaster = tf2_ros.TransformBroadcaster() 
        self.mask_topic=mask_topic
        self.pose_model=pose_model
        self.pose_refinement_model=pose_refinement_model
        self.minimum_num_pt=minimum_num_pt
        mask_sub=message_filters.Subscriber(self.mask_topic, mask)
        image_sub=message_filters.Subscriber(self.image_topic, Image)
        depth_sub=message_filters.Subscriber(self.depth_topic, Image)
        #depth_sub=rospy.Subscriber(self.depth_topic, Image,self.depth_callback)
        self.ts = message_filters.TimeSynchronizer([image_sub, mask_sub, depth_sub], 10)
        #self.pose_pub = rospy.Publisher(self.pose_, mask)
        #joint_e -1.33 joint_r -1.53
        self.height= 480
        self.width= 640
        self.ts.registerCallback(self.callback)
        self.depth_img=np.zeros((self.height,self.width))
        self.class_id_dict=class_id_dict

    def create_tf_message(self,pred_t, pred_r):
        transform_msg = TransformStamped()
        transform_msg.header.stamp = rospy.Time.now()  # Set the timestamp of the transform
        transform_msg.header.frame_id = 'rgbd_link_optical'  # Set the parent frame ID
        transform_msg.child_frame_id = 'object_frame'  # Set the child frame ID
        transform_msg.transform.translation.x = pred_t[0]  # Set the translation values (x, y, z)
        transform_msg.transform.translation.y = pred_t[1]
        transform_msg.transform.translation.z = pred_t[2]
        transform_msg.transform.rotation.x = pred_r[1]  # Set the rotation values (x, y, z, w)
        transform_msg.transform.rotation.y = pred_r[2]
        transform_msg.transform.rotation.z = pred_r[3]
        transform_msg.transform.rotation.w = pred_r[0]

        return transform_msg


    def callback(self,image_msg,mask_msg,depth_msg):
        
        if mask_msg.classes[0] != -10:
            mask_numpy=self.mask_to_numpy(mask_msg)
            depth_numpy=self.depth_callback(depth_msg)
            img_numpy=self.img_callback(image_msg)
            mask_depth = (depth_numpy != 0)

            img_numpy = np.array(img_numpy)[:, :, :3]
            img_numpy.astype(np.float32)
            seg_ids=[i for i in range(1,mask_numpy.shape[0]+1)]
            
            classes=list(np.array(mask_msg.classes)+1)
        
            img = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
            
            #for all objects in the scene
            for i in range(len(seg_ids)):
                #if the detected objects belong to the classes of interest perform pose estimation
                
                selected_obj_class= self.class_id_dict[classes[i]]
                seg_id=seg_ids[i]
                mask_label = mask_numpy[i,:,:]
            
                mask_label=mask_label.reshape(mask_label.shape[0],mask_label.shape[1])
                final_mask = mask_label * mask_depth
               
                if len(final_mask.nonzero()[0]) > self.minimum_num_pt:
                    cld_id=classes[i]
                    index=cld_id
                    rmin, rmax, cmin, cmax=get_bbox(mask_label)
                    img_masked = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]     
                
                    choose = final_mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
                    if len(choose) > self.num_points:
                        c_mask = np.zeros(len(choose), dtype=int)
                        c_mask[:self.num_points] = 1
                        np.random.shuffle(c_mask)
                        choose = choose[c_mask.nonzero()]
                    else:
                        choose = np.pad(choose, (0, self.num_points - len(choose)), 'wrap')

                    depth_masked = depth_numpy[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                    xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                    ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                    
                    choose = np.array([choose])
                    pt2 = depth_masked / self.cam_scale
                    pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
                    pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
                    cloud = np.concatenate((pt0, pt1, pt2), axis=1)

                    dellist = [k for k in range(0, len(cld[cld_id]))]
                    dellist = random.sample(dellist, len(cld[cld_id]) - num_points_mesh)
                    model_points = np.delete(cld[cld_id], dellist, axis=0)

                    with torch.no_grad():
                        model_points=torch.from_numpy(model_points.astype(np.float32))
                        cloud = torch.from_numpy(cloud.astype(np.float32))
                        choose = torch.LongTensor(choose.astype(np.int32))
                        img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
                        index = torch.LongTensor([int(cld_id) - 1])

                        cloud = cloud.view(1, num_points, 3)
                        choose = choose.view(1, 1, num_points)
                        model_points=model_points.view(1,num_points_mesh,3)
                        img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])
                        cloud = Variable(cloud).cuda()
                        choose = Variable(choose).cuda()
                        img_masked = Variable(img_masked).cuda()
                        index = Variable(index).cuda()
                        
                        pred_r, pred_t, pred_c, emb = self.pose_model(img_masked, cloud, choose, index)
                        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
                        pred_c = pred_c.view(bs, num_points)
                        pred_t = pred_t.view(bs * num_points, 1,3)
                        how_max, which_max = torch.max(pred_c, 1)
                        points = cloud.view(bs * num_points, 1, 3)
                        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
                        my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                        my_pred = np.append(my_r, my_t)

                        for ite in range(0, iteration):
                            T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
                            my_mat = quaternion_matrix(my_r)
                            R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                            my_mat[0:3, 3] = my_t
                            
                            new_cloud = torch.bmm((cloud - T), R).contiguous()
                            pred_r, pred_t = self.pose_refinement_model(new_cloud, emb, index)
                            pred_r = pred_r.view(1, 1, -1)
                            pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                            my_r_2 = pred_r.view(-1).cpu().data.numpy()
                            my_t_2 = pred_t.view(-1).cpu().data.numpy()
                            my_mat_2 = quaternion_matrix(my_r_2)
                            my_mat_2[0:3, 3] = my_t_2

                            my_mat_final = np.dot(my_mat, my_mat_2)
                            my_r_final = copy.deepcopy(my_mat_final)
                            my_r_final[0:3, 3] = 0
                            my_r_final = quaternion_from_matrix(my_r_final, True)
                            my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])
                            my_pred = np.append(my_r_final, my_t_final)
                            my_r = my_r_final
                            my_t = my_t_final

                    rospy.loginfo(f'pred={my_t},{my_r}')
                    tf_message=self.create_tf_message(my_t,my_r)
                    self.tf_broadcaster.sendTransform(tf_message)
                            
               
        else:
            rospy.loginfo("No object detected!")
        
            

    def img_callback(self,img_msg):
        img_numpy=  self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        return img_numpy

    def mask_to_numpy(self,objects_mask):
        # Reshape the received data into a 2D numpy array
        mask_data = np.array(objects_mask.mask.data, dtype=np.uint8)
        mask_data = mask_data.reshape((objects_mask.mask.layout.dim[2].size,objects_mask.mask.layout.dim[0].size, objects_mask.mask.layout.dim[1].size))

        return mask_data
     
    def depth_callback(self, depth_img):
        depth_data=  self.bridge.imgmsg_to_cv2(depth_img, desired_encoding="passthrough")
        depth_test=depth_data
        nan_mask = np.isnan(depth_data)
        depth_test=np.copy(depth_data)
        depth_test[nan_mask]=0
        return depth_test


if __name__ == '__main__':
    rospy.init_node('pose_estimation_model')
    pose_model_weight=rospy.get_param('~pose_model')
    pose_refine_model_weight=rospy.get_param('~pose_refine_model')
    classes= rospy.get_param('classes').split("\n")
    classes=[elem for elem in classes if len(elem) != 0]
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    cam_cx = rospy.get_param('~cx')
    cam_cy = rospy.get_param('~cy')
    cam_fx = rospy.get_param('~fx')
    cam_fy = rospy.get_param('~fy')
    cam_scale = rospy.get_param('~cam_scale')
    num_obj = len(classes)
    num_points = 1000
    num_points_mesh = 500
    minimum_num_pt= rospy.get_param('~minimum_num_pt')
    iteration = rospy.get_param('~refine_iteration')
    objects_classes=rospy.get_param('~objects_of_interest')
    bs = 1
    view_segmentation=rospy.get_param('~view_segmentation')

    pose_model = PoseNet(num_points = num_points, num_obj = num_obj)
    pose_model.cuda()
    pose_model.load_state_dict(torch.load(pose_model_weight))
    pose_model.eval()
    pose_refinement_model = PoseRefineNet(num_points = num_points, num_obj = num_obj)
    pose_refinement_model.cuda()
    pose_refinement_model.load_state_dict(torch.load(pose_refine_model_weight))
    pose_refinement_model.eval()

    class_id = 1
    cld = {}
    class_id_dict = {}
    base_path=rospy.get_param('~base_path')

    #load objects point clouds
    for elem in classes:
        #read the first line from the classes file
        class_input = elem
        #if the class on the current line doesn't belong to the list of class of interest skip to the next line
        if class_input in objects_classes:
            #else set a dictionary where the key is the class id and the value is class name
            class_id_dict[class_id]=class_input
            try:
                full_path=os.path.join(base_path,"YCB_Video_Models","models",class_input,"points.xyz")
                input_file=open(os.path.join(base_path,"YCB_Video_Models","models",class_input,"points.xyz"))
            except IOError as e:
                print(e)

            cld[class_id] = []
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1].split(' ')
                cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            input_file.close()
            cld[class_id] = np.array(cld[class_id])
            class_id += 1
    
    image_topic=rospy.get_param("image_topic_name")
    depth_topic=rospy.get_param("depth_topic_name")
    mask_topic=rospy.get_param("mask_topic_name")
    image_processor = ImageProcessor(image_topic,depth_topic,mask_topic,pose_model,pose_refinement_model,class_id_dict,num_points,minimum_num_pt,cam_scale)
    
   
    rospy.spin()
          
          
        

