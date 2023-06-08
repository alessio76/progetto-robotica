#!/usr/bin/env python3
from data import COLORS
from yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils import timer
from layers.output_utils import postprocess, undo_image_transformation
from data import cfg
from collections import defaultdict
import rospy
import os
from moto_project.msg import mask
import numpy as np
from PIL import Image
import cv2
import rospy
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


def prep_display(dets_out, img, h, w, args,undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
        
        if undo_transform:
            img_numpy = undo_image_transformation(img, w, h)
            img_gpu = torch.Tensor(img_numpy).cuda()
        else:
            img_gpu = img / 255.0
            h, w, _ = img.shape
        
        with timer.env('Postprocess'):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = postprocess(dets_out, w, h, visualize_lincomb = False,
                                            crop_masks        = False, #args.crop,
                                            score_threshold   = args.score_threshold)
            cfg.rescore_bbox = save

        with timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:args.top_k]
            
            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][idx]
            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
        num_dets_to_consider = min(args.top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < args.score_threshold:
                num_dets_to_consider = j
                break

            # Quick and dirty lambda for selecting the color for a particular index
            # Also keeps track of a per-gpu color cache for maximum speed
            def get_color(j, on_gpu=None):
                color_cache = defaultdict(lambda: {})
                color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
                
                if on_gpu is not None and color_idx in color_cache[on_gpu]:
                    return color_cache[on_gpu][color_idx]
                else:
                    color = COLORS[color_idx]
                    if not undo_transform:
                        # The image might come in as RGB or BRG, depending
                        color = (color[2], color[1], color[0])
                    if on_gpu is not None:
                        color = torch.Tensor(color).to(on_gpu).float() / 255.
                        color_cache[on_gpu][color_idx] = color
                    return color

        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
        if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
            # After this, mask is of size [num_dets, h, w, 1]
            masks = masks[:num_dets_to_consider, :, :, None]
            
            # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
            colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
            masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

            # This is 1 everywhere except for 1-mask_alpha where the mask is
            inv_alph_masks = masks * (-mask_alpha) + 1
            
            # I did the math for this on pen and paper. This whole block should be equivalent to:
            #    for j in range(num_dets_to_consider):
            #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
            masks_color_summand = masks_color[0]
            if num_dets_to_consider > 1:
                inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(dim=0)

            img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

        # Then draw the stuff that needs to be done on the cpu
        # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
        img_numpy = (img_gpu * 255).byte().cpu().numpy()

        if num_dets_to_consider == 0:
            return None,None,None
        
        if args.display_text or args.display_bboxes:
            for j in reversed(range(num_dets_to_consider)):
                x1, y1, x2, y2 = boxes[j, :]
                color = get_color(j)
                score = scores[j]
                if args.display_bboxes:
                    cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

                if args.display_text:
                    _class = cfg.dataset.class_names[classes[j]]
                    text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class

                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.6
                    font_thickness = 1

                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                    text_pt = (x1, y1 - 3)
                    text_color = [255, 255, 255]

                    cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                    cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
            
           
            return img_numpy,masks.cpu().numpy(),classes
           


class args:
    def __init__(self):
        self.top_k=5
        self.score_threshold=0.2
        self.display_masks=True
        self.display_bboxes=True
        self.display_text=True
        self.display_scores=True

class ImageProcessor:
    def __init__(self,image_topic,depth_topic,pose_model,pose_refinement_model,class_id_dict,num_points,yolact):
        self.bridge = CvBridge()
        self.depth_topic=depth_topic
        self.image_topic=image_topic
        self.num_points=num_points
        self.mask_topic=mask_topic
        self.pose_model=pose_model
        self.pose_refinement_model=pose_refinement_model
      
        image_sub=message_filters.Subscriber(self.image_topic, Image)
        depth_sub=message_filters.Subscriber(self.depth_topic, Image)
        #depth_sub=rospy.Subscriber(self.depth_topic, Image,self.depth_callback)
        self.ts = message_filters.TimeSynchronizer([image_sub,depth_sub], 10)
        #self.pose_pub = rospy.Publisher(self.pose_, mask)
        self.height= 480
        self.width= 640
        self.ts.registerCallback(self.callback)
        self.depth_img=np.zeros((self.height,self.width))
        self.class_id_dict=class_id_dict
        self.args=args()
        self.yolact=yolact
        self.height=480
        self.width=640

    def callback(self,image_msg,depth_msg):
        
        rospy.sleep(3)
      
        depth_numpy=self.depth_callback(depth_msg)
        img_numpy=self.img_callback(image_msg)
        mask_depth = (depth_numpy != 0)
        with torch.no_grad():
            ####YOLACT detection phase####
            frame = torch.from_numpy(np.asanyarray(img_numpy)[:,:,0:3]).cuda().float()
            batch = FastBaseTransform()(frame.unsqueeze(0))
            preds = self.yolact(batch)
        
            img_numpy,masks,classes = prep_display(preds, frame, None, None, undo_transform=False,args=self.args)
            rospy.loginfo(classes)
            #YOLACT classes start with 1, DenseFusion with 0 so increment the classes from YOLACT
            classes=list(np.array(classes)+1)
            img_numpy = np.array(img_numpy)[:, :, :3]
            img_numpy.astype(np.float32)
            """cv2.imshow("yolact detection", img_numpy)
            cv2.waitKey(0)"""
            
         
            seg_ids=[i for i in range(1,len(masks)+1)]
            print(masks.shape)
            #this because DenseFusion needs the image to be in the rgb format, while YOLACT is fine with bgr
            img = cv2.cvtColor(np.asanyarray(img_numpy), cv2.COLOR_BGR2RGB)
            #for all objects in the scene
            for i in range(len(seg_ids)):
                #if the detected objects belong to the classes of interest perform pose estimation
                
                selected_obj_class= class_id_dict[classes[i]]
                seg_id=seg_ids[i]
                mask_label = masks[i,:,:,0]
                mask_label=mask_label.reshape(mask_label.shape[0],mask_label.shape[1])
                mask = mask_label * mask_depth
                print(f'{mask.shape},{img_numpy.shape}')
                rospy.loginfo(f'len {len(mask.nonzero()[0])}')
                #perform pose estimation only if the object is visible and near enougth
                if len(mask.nonzero()[0]) > minimum_num_pt:
                    
                    cld_id=classes[i]
                    index=cld_id
                    rmin, rmax, cmin, cmax=get_bbox(mask_label)

                    img_masked = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]        
                    choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
                    if len(choose) > num_points:
                        c_mask = np.zeros(len(choose), dtype=int)
                        c_mask[:num_points] = 1
                        np.random.shuffle(c_mask)
                        choose = choose[c_mask.nonzero()]
                    else:
                        choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')
            
            """
            with torch.no_grad():

                    
                    #perform pose estimation only if the object is visible and near enougth
                    if len(mask.nonzero()[0]) > minimum_num_pt:
                        
                        cld_id=classes[i]
                        index=cld_id
                        rmin, rmax, cmin, cmax=get_bbox(mask_label)

                        img_masked = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]        
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
                    
                        
                        
                        pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
                        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
                        pred_c = pred_c.view(bs, num_points)
                        pred_t = pred_t.view(bs * num_points, 1,3)
                        how_max, which_max = torch.max(pred_c, 1)
                        points = cloud.view(bs * num_points, 1, 3)
                        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
                        #my_r[0]=np.absolute(my_r[0])
                        my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                        my_pred = np.append(my_r, my_t)
                        
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
                            # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)"""
                    
  
        
            

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
    torch.set_default_tensor_type('torch.cuda.FloatTensor')     
    weight_path=rospy.get_param('yolact_model')
    yolact = Yolact()
    yolact.load_weights(weight_path)
    yolact.eval()
    yolact.detect.use_fast_nms = True
    yolact.detect.use_cross_class_nms = True
    yolact.cuda()
    image_processor = ImageProcessor(image_topic,depth_topic,pose_model,pose_refinement_model,class_id_dict,num_points,yolact)
    
   
    rospy.spin()
          
          
        

