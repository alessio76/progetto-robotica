#!/usr/bin/env python3
from data import COLORS
from yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils import timer
from layers.output_utils import postprocess, undo_image_transformation
from data import cfg
import rospy
import warnings
from moto_project.msg import mask
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
import torch
from collections import defaultdict

warnings.filterwarnings("ignore")

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
    def __init__(self,yolact,segmentation_topic,image_topic_name,mask_topic):
        self.bridge = CvBridge()
        self.segmentation_topic=segmentation_topic
        self.image_topic=image_topic_name
        self.mask_topic=mask_topic
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.process_image)
        self.result_pub = rospy.Publisher(self.segmentation_topic, Image)
        self.mask_pub = rospy.Publisher(self.mask_topic, mask)
        self.args=args()
        self.yolact=yolact
        self.height=480
        self.width=640
        self.masks_msg=self.create_mask_msg(self.height,self.width)
        self.default_mask_msg=self.create_mask_msg(self.height,self.width)

        self.color=[[255, 255, 255], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0],
                [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128],
                [251, 194, 44], [240, 20, 134], [160, 103, 173], [70, 163, 210], [140, 227, 61],
                [128, 128, 0], [128, 0, 128], [0, 128, 128], [64, 0, 0], [0, 64, 0], [0, 0, 64]]
       

    def create_mask_msg(self,height_value,width_value):
        msg=mask()
        msg.mask.layout = MultiArrayLayout()
        height=MultiArrayDimension()
        width=MultiArrayDimension()
        object=MultiArrayDimension()
        height.label  = "height"
        height.size   = height_value
        width.label  = "width"
        width.size   = width_value
        width.stride =  width.size 
        height.stride = width.size * height.size
        object.label="object"
        object.stride=1
        object.size=1
        msg.mask.layout.dim=[height, width, object]
        msg.mask.data=np.zeros((1,height_value,width_value)).astype(np.int8).flatten()

        return msg
    
    def set_mask_data(self,msg,data,n_obj=1):
        #object
        msg.mask.layout.dim[2].size= n_obj
        msg.mask.layout.dim[2].stride= n_obj
        #height
        msg.mask.layout.dim[0].stride *= n_obj
        #width
        msg.mask.layout.dim[1].stride *= n_obj
        #data
        msg.mask.data = data.astype(np.int8).flatten()

    
    def process_image(self, img):
        # Convert the ROS Image message to OpenCV bgr format 
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        with torch.no_grad():
            frame = torch.from_numpy(cv_image).cuda().float()
            batch = FastBaseTransform()(frame.unsqueeze(0))
            preds = self.yolact(batch)
            processed_image,masks,classes=prep_display(preds, frame, None, None, self.args,undo_transform=False)

        # Convert the processed image back to a ROS Image message
        try:
            if processed_image is not None:
                result_img = self.bridge.cv2_to_imgmsg(processed_image,'bgr8')
                n_obj=masks.shape[0]
                self.set_mask_data(self.masks_msg,masks.astype(np.int8).flatten(),n_obj)
                out_mask_msg=self.masks_msg
                out_mask_msg.classes=classes
               
            else:
                out_mask_msg=self.default_mask_msg
                out_mask_msg.classes=[-10]
                result_img = self.bridge.cv2_to_imgmsg(cv_image,'bgr8')
          
            # Publish the result
            self.result_pub.publish(result_img)
            out_mask_msg.header.stamp=img.header.stamp
            out_mask_msg.header.seq=img.header.seq
            
            self.mask_pub.publish(out_mask_msg)
            
        except CvBridgeError as e:
            print(e)

        
        
if __name__ == '__main__':
    rospy.init_node('segmentation_node')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')     
    weight_path=rospy.get_param('yolact_model')
    yolact = Yolact()
    yolact.load_weights(weight_path)
    yolact.eval()
    yolact.detect.use_fast_nms = True
    yolact.detect.use_cross_class_nms = True
    yolact.cuda()
    segmentation_topic=rospy.get_param('segmentation_topic_name')
    mask_topic=rospy.get_param('mask_topic_name')
    image_topic_name=rospy.get_param('image_topic_name')
    image_processor = ImageProcessor(yolact,segmentation_topic,image_topic_name,mask_topic)
    rospy.spin()
