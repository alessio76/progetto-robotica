#!/bin/bash
if [[ $1 ]]
 then
python3 eval.py --display --cuda 1 --trained_model weights/yolact_resnet50_399_400000.pth --images $1:$2 --cross_class_nms 1 --score_threshold 0.5
else python3 eval.py --display --cuda 1 --trained_model weights/yolact_resnet50_399_400000.pth 
fi



