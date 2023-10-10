#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/my_train.py --dataset my_dataset\
  --dataset_root datasets/my_dataset\
  --start_epoch 116\
  --nepoch 20\
  --resume_posenet pose_model_49_0.0192243243732038.pth\
  --resume_refinenet pose_refine_model_116_0.01558202758204158.pth
