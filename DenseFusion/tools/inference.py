import _init_paths
import argparse
import os
import copy
import random
import numpy as np
from PIL import Image
import cv2
from data import COLORS, cfg
from yolact import Yolact
from utils.augmentations import  FastBaseTransform
from layers.output_utils import postprocess, undo_image_transformation
from utils import timer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import torch
import torch.nn.parallel
import glob
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import quaternion_matrix, quaternion_from_matrix
from collections import defaultdict

img_width = 480
img_length = 640
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
color_cache = defaultdict(lambda: {})
color = [[255, 255, 255], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0],
              [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128],
              [251, 194, 44], [240, 20, 134], [160, 103, 173], [70, 163, 210], [140, 227, 61],
              [128, 128, 0], [128, 0, 128], [0, 128, 128], [64, 0, 0], [0, 64, 0], [0, 0, 64]]

#converts string input fro boolean arguments to actual boolean
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
#extract classes, masks, bounding boxes and scroes from the YOLACT detection
def prep_display(dets_out, img, h, w, args,undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
 
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        #this function actually does the postprocess step. t is a vector containing the information
        #(see yolact_net/layers/output_utils for more informations)
        t = postprocess(dets_out, w, h, visualize_lincomb = False,
                                        crop_masks        = args.crop,
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
        global color_cache
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
    
    if args.display_fps:
            # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha


    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if args.display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    if num_dets_to_consider == 0:
        return img_numpy

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
            
    return img_numpy,boxes,masks.cpu().numpy(),classes


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
parser.add_argument('--model', type=str, default = 'pose_model_49_0.0192243243732038.pth',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = 'pose_refine_model_116_0.01558202758204158.pth',  help='resume PoseRefineNet model')
parser.add_argument('--objects_classes', nargs='+',help="for which class perform estimation")

# yolact args
parser.add_argument('--trained_model',default='/home/alessio/progetto_robotica/yolact_net/weights/yolact_resnet50_399_400000.pth', type=str,
                    help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to evaulate model')
parser.add_argument('--cross_class_nms', default=True, type=str2bool,
                    help='Whether compute NMS cross-class or per-class.')
parser.add_argument('--fast_nms', default=True, type=str2bool,
                    help='Whether to use a faster, but not entirely correct version of NMS.')
parser.add_argument('--display_masks', default=True, type=str2bool,
                    help='Whether or not to display masks over bounding boxes')
parser.add_argument('--display_bboxes', default=True, type=str2bool,
                    help='Whether or not to display bboxes around masks')
parser.add_argument('--display_text', default=True, type=str2bool,
                    help='Whether or not to display text (class [score])')
parser.add_argument('--display_scores', default=True, type=str2bool,
                    help='Whether or not to display scores in addition to classes')
parser.add_argument('--display', dest='display', action='store_true',
                    help='Display qualitative results instead of quantitative ones.')
parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                    help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                    help='The output file for coco bbox results if --coco_results is set.')
parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                    help='The output file for coco mask results if --coco_results is set.')
parser.add_argument('--seed', default=None, type=int,
                    help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                    help='Do not crop output masks with the predicted bounding box.')
parser.add_argument('--image', default=None, type=str,
                    help='A path to an image to use for display.')
parser.add_argument('--score_threshold', default=0.7, type=float,
                    help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')


parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False,
                    shuffle=False,
                    benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False,
                    display_fps=False,
                    emulate_playback=False)

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
#this high values make the selection of the point cloud points deterministic, so the pose is much more stable
#sometimes it swtiches to the opposite quaternion
#num_points=15000
num_points_mesh = 500
minimum_num_pt=2500
iteration = 2
bs = 1

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
class_id_dict = {}

#load objects point clouds
while 1:
    #read the first line from the classes file
    class_input = class_file.readline().replace('\n',"")
    #id there are no more classes end the cicle
    if not class_input:
        break
    #if the class on the current line doesn't belong to the list of class of interest skip to the next line
    if class_input in opt.objects_classes:
        #else set a dictionary where the key is the class id and the value is class name
        class_id_dict[class_id]=class_input
        input_file=open(os.path.join("datasets/my_dataset","YCB_Video_Models","models",class_input,"points.xyz"))
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

class_file.close()

"""
fig2= plt.figure()
ax2 = fig2.add_subplot(projection='3d')
img_plt = ax2.scatter(cld[1][:,0], cld[1][:,1], cld[1][:,2],c='g',label="estimated")

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
plt.title("original")
plt.show()"""

yolact = Yolact()
yolact.load_weights(opt.trained_model)
yolact.eval()
yolact.cuda()

torch.set_default_tensor_type('torch.cuda.FloatTensor')
#make sure that only one bounding box in selected, the one with highest confidence
yolact.detect.use_fast_nms = opt.fast_nms
yolact.detect.use_cross_class_nms = opt.cross_class_nms

#read rgb and depth
img = cv2.imread(opt.image)
depth_path = create_annotations_path(opt.image,'depth.exr')
depth = cv2.imread(depth_path,cv2.IMREAD_UNCHANGED)[:,:,0]
#since nvisii sets the background to -3*10e8 change that to 0
depth[depth<0]=0
#make background also the points too far from the camera so they are excluded
#this is necessary in particolar when we process background images where we have a rendered object (the background plane)
#located very far from the camera but still captured by the depth  
depth[depth>3]=0
#convert distance image into real depth
depth=depth_image_from_distance_image(depth,cam_fx,cam_fy,cam_cx,cam_cy)
mask_depth = (depth != 0)

with torch.no_grad():
    ####YOLACT detection phase####
    frame = torch.from_numpy(np.asanyarray(img)[:,:,0:3]).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = yolact(batch)
   
    img_numpy,boxes,masks,classes = prep_display(preds, frame, None, None, undo_transform=False,args=opt)
    #YOLACT classes start with 1, DenseFusion with 0 so increment the classes from YOLACT
    classes=list(np.array(classes)+1)
    img_numpy = np.array(img_numpy)[:, :, :3]
    img_numpy.astype(np.float32)
    """cv2.imshow("yolact detection", img_numpy)
    cv2.waitKey(0)"""
    
    my_result_wo_refine = []
    my_result = []
    seg_ids=[i for i in range(1,len(masks)+1)]
    print(masks.shape)
    #this because DenseFusion needs the image to be in the rgb format, while YOLACT is fine with bgr
    img = cv2.cvtColor(np.asanyarray(img), cv2.COLOR_BGR2RGB)
    #for all objects in the scene
    for i in range(len(seg_ids)):
        #if the detected objects belong to the classes of interest perform pose estimation
        
        selected_obj_class= class_id_dict[classes[i]]
        seg_id=seg_ids[i]
        mask_label = masks[i,:,:,0]
        mask_label=mask_label.reshape(mask_label.shape[0],mask_label.shape[1])
        mask = mask_label * mask_depth
        print(f'{mask.shape},{img_numpy.shape}')
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
                 # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)
            ##### for visualization
            #print(f'final my_t={my_t}')
            print(f'final my_r={my_r}\n')
            """my_r = quaternion_matrix(my_r)[:3, :3]
            dellist = [k for k in range(0, len(cld[cld_id]))]
            dellist = random.sample(dellist, len(cld[cld_id]) - num_points_mesh)
            model_points = np.delete(cld[cld_id], dellist, axis=0)
            model_points=torch.from_numpy(model_points.astype(np.float32))
            model_points=model_points.view(1,num_points_mesh,3)
            model_points = model_points[0].cpu().detach().numpy()
            pred = np.dot(model_points, my_r.T) + my_t
            fig1= plt.figure()
            ax1 = fig1.add_subplot(projection='3d')
            img_plt = ax1.scatter(pred[:,0], pred[:,1], pred[:,2],c='g',label="estimated")

            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.legend()
            title=opt.image.split('/')[-1]
            plt.title(title)
            plt.show()"""
           
          
        

