import _init_paths
import argparse
import os
import copy
import random
import numpy as np
from PIL import Image
import cv2
import json
import scipy.io as scio
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import scipy.misc
import numpy.ma as ma
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import glob
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix

img_width = 480
img_length = 640
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])

def get_imgs_paths(root_path):
    image_list=[]
    directories=[x for x in os.listdir(root_path) if os.path.isdir(os.path.join(root_path,x))]
    for dir in directories:
        full_path=str(os.path.join(root_path,dir))+'/'
        image_list.extend(glob.glob(full_path+'img/*.png'))

    return image_list

def depth_image_from_distance_image(distance, fx,fy,cx,cy):
  """Computes depth image from distance image.
  
  Background pixels have depth of 0
  
  Args:
      distance: HxW float array (meters)
      intrinsics: 3x3 float array
  
  Returns:
      z: HxW float array (meters)
  
  """
  
  height, width = distance.shape
  xlin = np.linspace(0, width - 1, width)
  ylin = np.linspace(0, height - 1, height)
  px, py = np.meshgrid(xlin, ylin)
  
  x_over_z = (px - cx) / fx
  y_over_z = (py - cy) / fy
  
  
  z = distance / np.sqrt(1. + x_over_z**2 + y_over_z**2)
  return z

#function to create the path for the annotation images or json file from 
#the image path
def create_annotations_path(path,png_replacement):
    #basically substitute png with whatever desidered value
    #json,depth.exr,seg.exr and img with annotations in the path 
    sub_dict={'img':'annotations','png':str(png_replacement)}
    temp=path

    for old,new in sub_dict.items():
        temp=temp.replace(old,new)
    return temp 

def get_bbox(label):
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

def str2bool(string):
    if string=='0' or string=='f' or string=='False' or string=='false':
        out=False
    else:
        out=True
    return out

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--dataset_root', type=str, default = '',  help='root dataset directory')
parser.add_argument('--image', type=str, default = '',  help='path to the input image')
parser.add_argument('--images', type=str, default = '',  help='path to the inout images directory')
parser.add_argument('--display', type=str, default = '0',  help='display point cloud')
opt = parser.parse_args()

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
opt.display=str2bool(opt.display)
cam_cx = 320
cam_cy = 240
cam_fx = 579.411376953125
cam_fy = 579.411376953125
cam_scale = 1
num_obj = 21
num_points = 1000
#num_points=15000
num_points_mesh = 500
minimum_num_pt=2500
iteration = 2
bs = 1
dataset_config_dir = 'datasets/my_dataset/dataset_config'
output_result_dir = 'experiments/eval_result/my_dataset'
result_wo_refine_dir = 'experiments/eval_result/my_dataset/Densefusion_wo_refine_result'
result_refine_dir = 'experiments/eval_result/my_dataset/Densefusion_iterative_result'

estimator = PoseNet(num_points = num_points, num_obj = num_obj)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()
refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj)
refiner.cuda()
refiner.load_state_dict(torch.load(opt.refine_model))
refiner.eval()

class_file = open('datasets/my_dataset/dataset_config/classes.txt')
class_id = 1
cld = {}
class_dict = {}
objects_classes=[]
while 1:
    class_input = class_input = class_file.readline().replace('\n',"")
    objects_classes.append(class_input)
    if not class_input:
        break
    class_dict[class_input]=class_id
    input_file=open(os.path.join(opt.dataset_root,"YCB_Video_Models","models",class_input,"points.xyz"))
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

img = Image.open(opt.image)

depth_path = create_annotations_path(opt.image,'depth.exr')
depth = cv2.imread(depth_path,cv2.IMREAD_UNCHANGED)[:,:,0]
#since nvisii sets the background to -3*10e8 change that to 0
depth[depth<0]=0
#convert distance image into real depth
depth=depth_image_from_distance_image(depth,cam_fx,cam_fy,cam_cx,cam_cy)
#make background also the points too far from the camera so they are excluded
#this is necessary in particolar when we process background images where we have a rendered object (the background plane)
#located very far from the camera but still captured by the depth  
depth[depth>3]=0
mask_depth = (depth != 0)
seg_path = create_annotations_path(opt.image,'seg.exr')

try:
    label = cv2.imread(seg_path,cv2.IMREAD_UNCHANGED)[:,:,0]
except:
    print(seg_path)
ann_path=create_annotations_path(opt.image,'json')
meta = json.load(open(ann_path))


my_result_wo_refine = []
my_result = []

for i,object in enumerate(meta['objects']):
    if object['class'] in objects_classes:
        seg_id=object['segmentation_id']
        mask_label = (label == seg_id)
        mask = mask_label * mask_depth
        if len(mask.nonzero()[0]) > minimum_num_pt:
        #read the name of the class and retrive the id
            selected_obj_class=meta['objects'][i]['class']
            cld_id=class_dict[selected_obj_class]
            index=cld_id
            rmin, rmax, cmin, cmax=get_bbox(mask_label)
            
            """cv2.imshow("mask_rgb",img[rmin:rmax,cmin:cmax,:])
            cv2.imshow("mask_depth",depth[rmin:rmax, cmin:cmax])
            cv2.waitKey()"""

            img_masked = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
            #orientation of the object frame in camera frame (xyzw format)
            target_r = np.array(meta['objects'][i]['quaternion_xyzw'])
            target_r= Rotation.from_quat(target_r).as_matrix()
            #position of the object frame in camera frame
            target_t = np.array(meta['objects'][i]['location'])
    
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) > num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')
        
            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            
            choose = np.array([choose])
            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)

            dellist = [k for k in range(0, len(cld[cld_id]))]
            dellist = random.sample(dellist, len(cld[cld_id]) - num_points_mesh)
            model_points = np.delete(cld[cld_id], dellist, axis=0)
            
            target = np.dot(model_points, target_r.T)+target_t
            
            model_points=torch.from_numpy(model_points.astype(np.float32))
            target=torch.from_numpy(target.astype(np.float32))
            cloud = torch.from_numpy(cloud.astype(np.float32))
            choose = torch.LongTensor(choose.astype(np.int32))
            img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
            index = torch.LongTensor([int(cld_id) - 1])

            cloud = cloud.view(1, num_points, 3)
            choose = choose.view(1, 1, num_points)
            model_points=model_points.view(1,num_points_mesh,3)
            target=target.view(1,num_points_mesh,3)
            img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])
            cloud = Variable(cloud).cuda()
            choose = Variable(choose).cuda()
            img_masked = Variable(img_masked).cuda()
            index = Variable(index).cuda()
          
            
            
            pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
            pred_c = pred_c.view(bs, num_points)
            pred_t = pred_t.view(bs * num_points, 1,3)
            how_max, which_max = torch.max(pred_c, 1)
            points = cloud.view(bs * num_points, 1, 3)
            my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
            my_pred = np.append(my_r, my_t)
            
            """
            choose_list.append(torch.LongTensor(choose.astype(np.int32)))
            img_list.append(norm(torch.from_numpy(img_obj.astype(np.float32))))
            target_list.append(torch.from_numpy(target.astype(np.float32)))
            model_points_list.append(torch.from_numpy(model_points.astype(np.float32)))
            class_id_list.append(torch.LongTensor([int(cld_id)-1]))"""
            my_result_wo_refine.append(my_pred.tolist())
            #print(f'initial my_t={my_t}')
            print(f'initial my_r={my_r}')
            for ite in range(0, iteration):
                T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
            
                my_mat = quaternion_matrix(my_r)
                R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                my_mat[0:3, 3] = my_t
                
                new_cloud = torch.bmm((cloud - T), R).contiguous()
                pred_r, pred_t = refiner(new_cloud, emb, index)
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
            # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)
            my_result.append(my_pred.tolist())
            #print(f'my_t={my_t}')
            print(f'my_r={my_r}\n')
            model_points = model_points[0].cpu().detach().numpy()
            my_r = quaternion_matrix(my_r)[:3, :3]
            pred = np.dot(model_points, my_r.T) + my_t
            target = target[0].cpu().detach().numpy()

            dis = np.mean(np.linalg.norm(pred - target, axis=1))

            if dis < 0.02:
                print('No.{0} Pass! Distance: {1}'.format(i, dis))
            else:
                print('No.{0} NOT Pass! Distance: {1}'.format(i, dis))

            if opt.display:
                ##### for visualization
                fig1= plt.figure()
                ax1 = fig1.add_subplot(projection='3d')
                img_plt = ax1.scatter(target[:,0], target[:,1], target[:,2],c='r',label="gt")

                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_zlabel('Z')
                #ax1 = fig1.add_subplot(projection='3d')
                img_plt = ax1.scatter(pred[:,0], pred[:,1], pred[:,2],c='g',label="estimated")

                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_zlabel('Z')
                ax1.legend()
                title=opt.image.split('/')[-1]
                plt.title(title)
                plt.show()

