import json
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import time
from itertools import permutations

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]

def get_bbox_ycb(label):
    t=time.time()
    img_width=480
    img_length=640
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
    print("ycb",(time.time()-t)*1e6)
    return rmin, rmax, cmin, cmax

def get_bbox(bbx):
    t=time.time()
    img_width = 480
    img_length = 640
    """if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639"""               
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
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
    print("altro",(time.time()-t)*1e6)
    return rmin, rmax, cmin, cmax


test_len=5
img_list=[]
ann_list=[]
obj_list=['002_master_chef_can']
minimum_num_pt=0
num_pt=1000
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])

height= 480

     
cam_cx=320.0
cam_cy= 240.0
cam_fx= 579.411376953125
cam_fy= 579.411376953125

#for i in range(2):
for i,file in enumerate(glob.glob("annotations/0*.json")):
    #print("neww_image")
    img_path=os.path.join("img",'%05d.png' % i)
    img_rgb=cv2.imread(img_path)
    cv2.imshow("original",img_rgb)
    depth_path=os.path.join("annotations",'%05d.depth.exr' % i)
    depth=cv2.imread(depth_path,cv2.IMREAD_UNCHANGED)[:,:,0]
    depth[depth < -10]=0
    seg_path=os.path.join("annotations",'%05d.seg.exr' % i)
    label=cv2.imread(seg_path)[:,:,0]
    ann_path=os.path.join("annotations",'%05d.json' % i)
    meta=json.load(open(ann_path))
    
    for j,object in enumerate(meta['objects']):
            if object['class'] in obj_list:
                seg_id=object['segmentation_id']
                mask_depth = (depth != 0)
                mask_label = (label == seg_id)
                mask = mask_label * mask_depth
                if len(mask.nonzero()[0]) > minimum_num_pt:
                    obj_id=j
                    selected_obj_class=object['class']
                    bbox=object['bounding_box_minx_maxx_miny_maxy']
                    rmin, rmax, cmin, cmax=get_bbox([bbox[2],bbox[3],bbox[0],bbox[1]])
                    rmin_ycb, rmax_ycb, cmin_ycb, cmax_ycb=get_bbox_ycb(mask_label)
                    
                    index_perms=list(permutations([0,1,2,3]))
                    
                    """
                    for i,perm in enumerate(index_perms):
                        new_box=list((bbox[perm[0]],bbox[perm[1]],bbox[perm[2]],bbox[perm[3]]))
                        rmin, rmax, cmin, cmax=get_bbox(new_box)
                        result=np.array((rmin_ycb-rmin,rmax_ycb-rmax,cmin_ycb-cmin,cmax_ycb-cmax))
                        if (result[0]==0 and result[1]==0) or (result[2]==0 and result[3]==0):
                            print("####")
                            print(rmin_ycb-rmin,rmax_ycb-rmax,cmin_ycb-cmin,cmax_ycb-cmax)
                            print(perm)
                            print("####")
                            print('\n')
                    """

                    img = np.transpose(np.array(img_rgb)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
                    img_ycb = np.transpose(np.array(img_rgb)[:, :, :3], (2, 0, 1))[:, rmin_ycb:rmax_ycb, cmin_ycb:cmax_ycb]
                    try:
                        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
                        if len(choose) > num_pt:
                            c_mask = np.zeros(len(choose), dtype=int)
                            c_mask[:num_pt] = 1
                            np.random.shuffle(c_mask)
                            choose = choose[c_mask.nonzero()]
                        else:
                            choose = np.pad(choose, (0, num_pt - len(choose)), 'wrap')
                    except ValueError as e:
                        print(rmin_ycb-rmin,rmax_ycb-rmax,cmin_ycb-cmin,cmax_ycb-cmax)
                        print(bbox[2],bbox[3],bbox[0],bbox[1])
                        print(file)
                        print("error")

                    depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                    xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                    ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                    
                    choose = np.array([choose])
                    cam_scale = 1
                    pt2 = depth_masked / cam_scale
                    pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
                    pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
                    cloud = np.concatenate((pt0, pt1, pt2), axis=1)
"""
                    cv2.imshow(f"obj {j}",img_rgb[rmin:rmax,cmin:cmax,:])
                    cv2.imshow(f"obj_ycb {j}",img_rgb[rmin_ycb:rmax_ycb,cmin_ycb:cmax_ycb,:])
                    cv2.imshow(f"depth{j}",depth[rmin:rmax,cmin:cmax])
                    cv2.imshow(f"depth_ycb {j}",depth[rmin_ycb:rmax_ycb,cmin_ycb:cmax_ycb])
                    cv2.waitKey(0)
                    cv2.destroyWindow(f"obj {j}")
                    cv2.destroyWindow(f"obj_ycb {j}")
                    cv2.destroyWindow(f"depth{j}")
                    cv2.destroyWindow(f"depth_ycb {j}")"""
            
    #cv2.destroyWindow("original")                 