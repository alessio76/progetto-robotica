import json 
import os
import glob
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
#the enviromental variable must be set before importing cv2 otherwise nothing works
import cv2
import argparse
import numpy as np
from numpy import ma
import time

parser = argparse.ArgumentParser()

parser.add_argument(
    '--train-ann-directory',
    default="train/annotations",
    help = 'train annotations directory path'
)

parser.add_argument(
    '--val-ann-directory',
    default="val/annotations",
    help = 'val annotations directory path'
)

parser.add_argument(
    '--out-train-file',
    default='train.json',
    help = 'output coco annotations for train set'
)

parser.add_argument(
    '--out-val-file',
    default='val.json',
    help = 'output coco annotations for validation set'
)

parser.add_argument(
    '--class-file',
    default='/home/alessio/progetto_robotica/src/yolact/data/my_dataset/classes.txt',
    help = 'categories file'
)


opt = parser.parse_args()
image_dict=[]
category_dict=[]
annotations_dict=[]
categories=[]
coco_coordinates=[] 
#outmost keys
keys=("images","categories","annotations")
#define the keys for image subelement
images_keys=('id','width','height',"file_name","coco_url")
#define the keys for annotation subelement
annotation_keys=('segmentation','area','iscrowd','image_id','bbox','category_id',"id")
supercategory="YCB_Video"
nvsii_to_class_names={'Caffe':None}
#read class names
with open(opt.class_file) as class_file:
    categories=class_file.read()
    categories=categories.split("\n")
    categories.pop(-1)
    
start_time=time.time()
#create the category entries in the json file
for i,category in enumerate(categories,start=1):
    category_dict.append({"supercategory": supercategory, "id": i, "name": category})
    if category=='002_master_chef_can':
        nvsii_to_class_names['Caffe']=i

info_list=[]
if opt.train_ann_directory is not None:
#structure to store directory and file info for train and validation
	train_dict={'input_directory':opt.train_ann_directory, 'out_file':opt.out_train_file}
	info_list.append(train_dict)

if opt.val_ann_directory is not None:
	val_dict={'input_directory':opt.val_ann_directory, 'out_file':opt.out_val_file}
	info_list.append(val_dict)


for info in info_list:
	#directory where all the nvisii data are stored
	json_file_names=glob.glob(os.path.join(info['input_directory'],"0*.json"))
	num_objects=0
	#json file outputted by nvisii
	for i,json_file_name in enumerate(json_file_names,start=1):
		with open(json_file_name) as json_file:
			data = json.load(json_file)
			#create the image sublement taking the informations from each
			coco_url=os.path.join(os.path.abspath(os.getcwd()),json_file_name.replace(".json",".png"))
			coco_url=coco_url.replace("annotations","img")
			file_name=json_file_name.replace(f"{info['input_directory']}/","")
			image_dict.append(dict(zip(images_keys,(i,data['camera_data']['width'],data['camera_data']['height'],file_name.replace(".json",".png"),coco_url))))
			
		#for every object in the image
		for object_dict in data['objects']:
			#calculate boundaries for each object
			seg=cv2.imread(os.path.join(info['input_directory'],file_name.split(".")[0]+".seg.exr"),cv2.IMREAD_GRAYSCALE)
			#in seg each mask is filled with an integer which is the id of the specifc object, so first
			#read the id of that object from the json input file and then use that id to identify the mask of that object
			seg_id=object_dict['segmentation_id']
			obj_mask=(seg == seg_id).astype(np.uint8)*255
			#cv2 function to calculate contour of the object
			contours= cv2.findContours(obj_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
			contours_rows=np.array([contour.shape[0] for contour in contours])

			#add an annotation only if the border of the object has more than 2 points 
			#so check if there is at least one False element in the boolean array that checks if the contour points >2
			#(the number of points is equal the number of rows)
			if object_dict['class'] in categories and False not in (contours_rows>2):
				num_objects+=1
				coco_coordinates=[]
				#reorder bbox coordinate in the order expected by COCO
				top_left_x = object_dict['bounding_box_minx_maxx_miny_maxy'][0]
				top_left_y = object_dict['bounding_box_minx_maxx_miny_maxy'][2]
				bottom_right_x = object_dict['bounding_box_minx_maxx_miny_maxy'][1]
				bottom_rigth_y = object_dict['bounding_box_minx_maxx_miny_maxy'][3]
				rect_width = bottom_right_x -  top_left_x
				rect_height = bottom_rigth_y - top_left_y 
				bbox=(top_left_x,top_left_y,rect_width,rect_height)
				
				#convert numpy array into a list in the form (x,y) in pixels
				#it is a list of list because if occluded the segmentaton mask could be split in different pieces
				for border in contours:
					rows=border.shape[0]
					coco_coordinates.append(border.reshape(rows*2).tolist())

				#area is simply the number of visible pixels
				area=len(np.where(seg == seg_id)[0])
				iscrowd=0
				#create the annotations subelement
				annotations_dict.append(dict(zip(annotation_keys,(coco_coordinates,area,iscrowd,i,bbox,nvsii_to_class_names['Caffe'],num_objects))))

				
	#put all togheter the dict for the subelement to form the final json			
	final_dict=dict(zip(keys,(image_dict,category_dict,annotations_dict)))
	with open(info['out_file'], 'w') as outfile:   
		#outfile.write(json.dumps(final_dict,indent=4,sort_keys=True))
		json.dump(final_dict,outfile)
		final_dict.clear()
		annotations_dict.clear()
		image_dict.clear()
		
			

print("time",time.time()-start_time)

	


