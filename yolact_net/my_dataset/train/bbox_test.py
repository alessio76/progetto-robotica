import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

""" Convert a mask to border image """
def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

""" Mask to bounding boxes """
def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes
coco=COCO('../train.json')

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))


catIds = coco.getCatIds(catNms=['002_master_chef_can'])
imgIds = coco.getImgIds(catIds=catIds)
img_ann=coco.loadImgs(imgIds)
fig=plt.figure()
ax=fig.add_subplot(2,1,1)
x=cv2.imread(img_ann[1]['coco_url'])
ax.imshow(x)
plt.axis('off')
annIds = coco.getAnnIds(imgIds=img_ann[1]['id'], iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns,draw_bbox=True)


y_path=img_ann[1]['coco_url'].replace("img","annotations")
y_path=y_path.replace(".png",".seg.exr")
y=cv2.imread(y_path,cv2.IMREAD_GRAYSCALE)
print(np.unique(y))
print(img_ann[1]['coco_url'])
for i in range(6):
	array=(y==i).astype(int)
	bboxes = mask_to_bbox((array*255))
	print(len(bboxes))
	""" marking bounding box on image """
	for bbox in bboxes:
		x = cv2.rectangle(x, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
    
ax2=fig.add_subplot(2,1,2)
ax2.imshow(x)
plt.show()


