#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 train.py --config=my_yolact_ycb_config  --resume=weights/yolact_resnet50_192_192027_interrupt.pth --batch_size 2 --cuda 1 --validation_size 500
