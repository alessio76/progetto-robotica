from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import argparse
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

def show_images(n_img_to_show,imgIds):
    
    img_ann=coco.loadImgs(imgIds)
    n_img=len(img_ann)
    print(n_img)
    count=0
    result,rest = (n_img // n_img_to_show, n_img % n_img_to_show)

    for k in range(result+rest):
        if count+n_img_to_show<n_img:
            img_to_load=n_img_to_show
        else:
            img_to_load=n_img-count
        fig=plt.figure()
        for i in range(count,count+img_to_load):
            ax =fig.add_subplot(n_img_to_show,1,i+1-count)
            I = Image.open(img_ann[i]['coco_url'])
            ax.imshow(I)
            plt.axis('off')
            annIds = coco.getAnnIds(imgIds=imgIds[i], iscrowd=None)
            anns = coco.loadAnns(annIds)
            coco.showAnns(anns,draw_bbox=True)
        
        plt.show()
        count+=img_to_load

def sanity_check(imgIds):
    annIds=coco.getAnnIds(imgIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    for ann in anns:
       assert len(ann['segmentation']) >0," segmentation is empty"
       for elem in ann['segmentation']:
           assert len(elem)>2, "number of border points <2"

parser = argparse.ArgumentParser()

parser.add_argument(
    '--img-to-show',
    default=3,
    type=int,
    help = 'number of images to show in a iteration'
)

parser.add_argument(
    '--ann-file',
    default=None,
    help = 'annotation file to load'
)

parser.add_argument(
    '--show-img',
    default=False,
    dest='show_img',
    action='store_true',
    help = 'show annotations on images'
)

opt = parser.parse_args()
n_img_to_show=opt.img_to_show
annFile=opt.ann_file

coco=COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))
catIds = coco.getCatIds(catNms=['002_master_chef_can'])
imgIds = coco.getImgIds(catIds=catIds)

#check if the annotations are properly filled
sanity_check(imgIds)

#shows image annotations
if opt.show_img:
    show_images(opt.img_to_show,imgIds)


   
    
            




