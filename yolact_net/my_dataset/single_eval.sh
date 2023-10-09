#!/bin/bash
python3 -W "ignore" yolact_test.py --display --cuda 1 --trained_model weights/yolact_resnet50_399_400000.pth --image $1 --cross_class_nms 1 
