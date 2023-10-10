#!/bin/bash
#-x prints the commands that are executed,+x suppress it
set +x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 -W ignore ./tools/test_single_image.py --dataset_root datasets/my_dataset\
  --model trained_models/my_dataset/pose_model_49_0.0192243243732038.pth\
  --refine_model trained_models/my_dataset/pose_refine_model_116_0.01558202758204158.pth\
  --image test/00005.png\
  --display 1
  #--image datasets/my_dataset/val/2/img/00613.png

: '
vector=($(find /home/alessio/progetto_robotica/src/DenseFusion/test -maxdepth 1 -type f -name "*.png"))

# Iterate over the elements of the vector
for element in "${vector[@]}"
do
    # Call dense_fusion_test.py with the --image parameter
    python3 -W ignore ./tools/inference.py --dataset_root datasets/my_dataset\
	  --model trained_models/my_dataset/pose_model_49_0.0192243243732038.pth\
	  --display 0\
	  --refine_model trained_models/my_dataset/pose_refine_model_116_0.01558202758204158.pth\
	  --image "$element"
done
'
  
