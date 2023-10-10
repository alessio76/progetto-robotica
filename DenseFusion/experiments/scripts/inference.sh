#!/bin/bash
#makes only pose prediction
set +x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0
image_path="datasets/my_dataset/train/2/img/02069.png"
python3 -W ignore ./tools/inference.py --model trained_models/my_dataset/pose_model_49_0.0192243243732038.pth\
  --image $image_path\
  --refine_model trained_models/my_dataset/pose_refine_model_116_0.01558202758204158.pth\
  --objects_classes 002_master_chef_can
   #--image datasets/my_dataset/val/2/img/00613.png\
