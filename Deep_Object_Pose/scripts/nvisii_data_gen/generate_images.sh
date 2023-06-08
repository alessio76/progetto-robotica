#!/bin/bash


#path="/home/alessio/progetto_robotica/src/DenseFusion/datasets/my_dataset/val/6"
script_path="/home/alessio/progetto_robotica/src/DenseFusion/datasets/my_dataset/val/4/postprocess.sh"
#path="output/output_example"
path="/home/alessio/progetto_robotica/src/DenseFusion/test"
python3 my_dataset_generator.py --spp 1500 --nb_frames 20 --scale 1 --outf $path --width 640 --height 480 --nb_distractors 0 --nb_objects 1 --start_image_number 10  --skip_frame 200 --z_max 0.2 --y_max 0.2 --force_max 2 --script_path $script_path #--bullet_gui 1

#intrisc corresponds to asus-uw.json  --fx 1066.778 --fy 1076.487 --cx 312.9869 --cy 241.3109 

