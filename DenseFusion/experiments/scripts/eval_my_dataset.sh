#!/bin/bash
#evaluate my dataset

set +x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 -W ignore ./tools/eval_my_dataset.py --dataset_root datasets/my_dataset\
  --model trained_models/my_dataset/pose_model_49_0.0192243243732038.pth\
  --refine_model trained_models/my_dataset/pose_refine_model_116_0.01558202758204158.pth\
  --calculate_AUC 1 #set to 0 to calculate only the ADD>2cm metric
  #--calculate_ADD 1
