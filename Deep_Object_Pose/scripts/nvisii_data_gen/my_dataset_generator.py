#this transform make the enviroment look "upright", otherwise it will be upside down when the base world frame in x rigth,y down and z forward
#***DON'T ELIMINATE***
#q1=visii.angleAxis(-visii.pi()/2,visii.vec3(1,0,0))
#***DON'T ELIMINATE***

import os
import glob
import pybullet as p
import time
import subprocess
import nvisii as visii
import numpy as np
import random
import argparse
from utils import *
from my_utils import *
######################## COMMAND LINE ARGUMENTS
parser = argparse.ArgumentParser()

parser.add_argument(
    '--spp',
    default=800,
    type=int,
    help = "number of sample per pixel, higher the more costly"
)

parser.add_argument(
    '--initial_random_shift',
    default=True,
    type=bool,
    help = "If True add a random shift to the initial object position"
)

parser.add_argument(
    '--script_path',
    help = "path to the script to create the directory structure"
)

parser.add_argument(
    '--width',
    default=640,
    type=int,
    help = 'image output width'
)
parser.add_argument(
    '--height',
    default=480,
    type=int,
    help = 'image output height'
)
# TODO: change for an array
parser.add_argument(
    '--objs_folder_distrators',
    default='google_scanned_models/',
    help = "object to load folder"
)
parser.add_argument(
    '--objs_folder',
    default='models/',
    help = "object to load folder"
)

parser.add_argument(
    '--scale',
    default=1,
    type=float,
    help='Specify the scale of the target object(s). If the obj mesh is in '
         'meters -> scale=1; if it is in cm -> scale=0.01.'
)

parser.add_argument(
    '--distractor_scale',
    default=2,
    type=float,
    help='Specify the scale of the target object(s). If the obj mesh is in '
         'meters -> scale=1; if it is in cm -> scale=0.01.'
)

parser.add_argument(
    '--skyboxes_folder',
    default='dome_hdri_haven/',
    help = "dome light hdr"
)

parser.add_argument(
    '--nb_objects',
    default=1,
    type = int,
    help = "how many objects"
)
parser.add_argument(
    '--nb_distractors',
    default=1,
    help = "how many objects"
)
parser.add_argument(
    '--nb_frames',
    default=2000,
    help = "how many frames to save"
)
parser.add_argument(
    '--skip_frame',
    default=100,
    type=int,
    help = "how many frames to skip"
)
parser.add_argument(
    '--noise',
    action='store_true',
    default=False,
    help = "if added the output of the ray tracing is not sent to optix's denoiser"
)
parser.add_argument(
    '--outf',
    default='output_example/',
    help = "output filename inside output/"
)
parser.add_argument('--seed',
    default = None,
    help = 'seed for random selection'
)

parser.add_argument(
    '--motionblur',
    action='store_true',
    default=False,
    help = "use motion blur to generate images"
)

parser.add_argument(
    '--focal-length',
    default=None,
    type=float,
    help = "focal length of the camera"
)

parser.add_argument(
    '--fx',
    default=None,
    type=float,
    help = "fx intrisic parameter of the camera"
)

parser.add_argument(
    '--fy',
    default=None,
    type=float,
    help = "fy intrisic parameter of the camera"
)

parser.add_argument(
    '--cx',
    default=None,
    type=float,
    help = "cx intrisic parameter of the camera"
)

parser.add_argument(
    '--cy',
    default=None,
    type=float,
    help = "cy intrisic parameter of the camera"
)

parser.add_argument(
    '--bullet_gui',
    default=False,
    type=bool,
    help = "If True shows the bullet gui"
)

parser.add_argument(
    '--visibility-fraction',
    action='store_true',
    default=False,
    help = "Compute the fraction of visible pixels and store it in the "
           "`visibility` field of the json output. Without this argument, "
           "`visibility` is always set to 1. Slows down rendering by about "
           "50 %%, depending on the number of visible objects."
)

parser.add_argument(
    '--debug',
    action='store_true',
    default=False,
    help="Render the cuboid corners as small spheres. Only for debugging purposes, do not use for training!"
)

parser.add_argument(
    '--start_image_number',
    default=0,
    type=int,
    help = 'start number to name images'
)

parser.add_argument(
    '--y_max',
    default=0.4,
    type=float,
    help = 'y position of the limiting plane perpendicular to the y axis'
)

parser.add_argument(
    '--z_max',
    default=0.4,
    type=float,
    help = 'z position of the limiting plane perpendicular to the z axis'
)

parser.add_argument(
    '--force_max',
    default=10,
    type=float,
    help = 'range where the force can randomly vary. The values will be drawned from (-force_max,force_max)'
)

parser.add_argument(
    '--limiting_box',
    default=True,
    type=bool,
    help = 'If True, create the physical limiting box that costrains objects movement'
)

opt = parser.parse_args()

# # # # # # # # # # # # # # # # # # # # # # # # #

if os.path.isdir(f'{opt.outf}'):
    print(f'folder {opt.outf}/ exists')
else:
    os.makedirs(f'{opt.outf}')
    print(f'created folder {opt.outf}/')

opt.outf = f'{opt.outf}'

if not opt.seed is None:
    random.seed(int(opt.seed))

# # # # # # # # # # # # # # # # # # # # # # # # #
if opt.bullet_gui:
    physicsClient  = p.connect(p.GUI)  
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING,1)
else:
    physicsClient = p.connect(p.DIRECT) # non-graphical version
# # # # # # # # # # # # # # # # # # # # # # # # #
start_time=time.time()

visii.initialize(headless = True, lazy_updates = False)

if not opt.motionblur:
    visii.sample_time_interval((1,1))

visii.sample_pixel_area(
    x_sample_interval = (.5,.5),
    y_sample_interval = (.5, .5))


if not opt.noise:
    visii.enable_denoiser()

#world frame and nvisii bullet: x forward, y left, z up 

#max and min distances from the camera long the view axis, which is relative to world frame the x axis
far=2
near=0.1

#initial object position
obj_base_position=visii.vec3(0.4,0,0)

#define camera frame x right, y down and z forward respect to pybullet world frame
Tw_c_bullet=np.zeros((4,4))
Tw_c_bullet[3,3]=1
xc_bullet=(0,-1,0)
yc_bullet=(0,0,-1)
zc_bullet=(1,0,0)
#transponse to write column-wise 
Tw_c_bullet[:3,:3]=np.array((xc_bullet,yc_bullet,zc_bullet)).T

#create nvisii and pybullet camera
#1 argument=camera frame relative to world in bullet for visualization
camera=create_camera(Tw_c_bullet,fx=opt.fx,fy=opt.fy,cx=opt.cx,cy=opt.cy, width=opt.width,height=opt.height)

#'at' is the forward vector, 'up' points up, 'eye' the position of the camera (in world frame)
#this means x forward, y left, z up, same as in bullet; defines the world nvisii frame
"IMPORTANT: if you run get_world_forward() you get the opposite x vector because the forward vector is oriented from the screen to the viewer"

camera_struct = {
    'at':visii.vec3(1,0,0),
    'up':visii.vec3(0,0,1),
    'eye':visii.vec3(0,0,0)
}

camera.get_transform().look_at(at=camera_struct['at'],eye=camera_struct['eye'],up=camera_struct['up'])
visii.set_camera_entity(camera)

# # # # # # # # # # # # # # # # # # # # # # # # #
#plane positions in camera frame
y_lim=opt.y_max*np.array((-1,1))
z_lim=opt.z_max*np.array((-1,1))

#plane positions
positions=(far,near,float(y_lim[0]),float(y_lim[1]),float(z_lim[0]),float(z_lim[1]))

#orientation to make the plane perpendiculat to the axes and form a box
orientations=(Rotation.from_euler('y', 90, degrees=True).as_quat(),
              Rotation.from_euler('x', 90, degrees=True).as_quat(),
              Rotation.from_quat((0,0,0,1)))
axes=np.eye(3)
j=0

for i,pos in enumerate(positions):
    if i %2 ==0 and i>0:
        j+=1
    create_limiting_plane(axes[j,:]*np.array(pos),orientations[j])


# # # # # # # # # # # # # # # # # # # # # # # # #

###### OBJECTS FOLDER PARAMETERS
objects_dict={
    'folder' : glob.glob(opt.objs_folder + "*/"),
    'mesh_path' : "/google_16k/textured.obj",
    'textures_path' : "/google_16k/texture_map_flat.png",
    'name_prefix' : "hope_",
    'n_objects' : int(opt.nb_objects),
    'scale':opt.scale
}

###### DISTRACTORS FOLDER PARAMETERS
distractors_dict={
    'folder' :  glob.glob(opt.objs_folder_distrators + "*/"),
    'mesh_path' : "/meshes/model.obj",
    'textures_path' : "/materials/textures/texture.png",
    'name_prefix' : "google_",
    'n_objects' : int(opt.nb_distractors),
    'scale':opt.distractor_scale
}

parameter_list=[objects_dict,distractors_dict]
visii_pybullet=[]
names_to_export=[]
#load objects and distractors
for parameters in parameter_list:
    if parameters['n_objects']>0:
       #make a random shift to scatter a bit the objects from the common initial position
       if opt.initial_random_shift:
        random_shift=visii.vec3(random.uniform(-0.1,0.5),random.uniform(-0.1,0.1),random.uniform(-0.1,0.1))

       temp_obj,temp_names=load_objects(parameters,obj_base_position + random_shift,opt.debug)
       visii_pybullet.extend(temp_obj)
       names_to_export.extend(temp_names)
    else:
        print("n_objects is 0")

export_to_ndds_folder_settings_files(
    opt.outf,
    obj_names=names_to_export,
    width=opt.width,
    height=opt.height,
    camera_name='camera',
)

#counter for the rendered frames
i_render = 0
#counter for the generated frames
i_frame=0
#index in the background vector
i_background=0
backgrounds = glob.glob(f'{opt.skyboxes_folder}/*.hdr')
background_colors={'red':(255,0,0),'magenta':(255,0,255),'green':(0,255,0),
                   'gray':(100,100,100)}

backgrounds.extend(background_colors.keys())
n_backgrounds=len(backgrounds)
n_frames_per_background = int(opt.nb_frames)//n_backgrounds
print(f'{n_frames_per_background} frames per background')
skybox_name='dome_tex'

#range in which each component can vary
force_max=np.ones(3)*opt.force_max
#this flag checks if the uniform background exists (1) ot it
#must be created (0)
flag=0

#select the offset to add to the names of the iages generated
start_image_number=opt.start_image_number
#NVISII QUATERNIONS HAVE W,X,Y,Z FORMAT, WHILE IN THE JSON THE ARE WRITTEN AS X,Y,Z,W
#BULLET QUATERNIONS ARE IN X,Y,Z,W FORMAT
while i_render<int(opt.nb_frames):
    #do a simulation step so apply force and update position
    p.stepSimulation()
    i_frame+=1
    #skip some frame to move the objects randomly
    #take only few frames to scatter the objects in the scene
    if i_frame % int(opt.skip_frame)==0:
        if backgrounds[i_background].endswith('.hdr'):
            #generate images with the .hdr backgrounds
            dome_tex=set_skybox(single_skybox_path=backgrounds[i_background],skybox_name=skybox_name)
        elif flag==0:
            #generate images with a uniform background
            #if it's the first uniform background image create the background 
            skybox_time=time.time()
            flag=1
            color=backgrounds[i_background]
            corrected_color=np.array(background_colors[color])/255
            #place the plane wich acts as background very far from the scene
            plane_distance=10
            background_material=set_uniform_background(corrected_color,plane_distance)
            opt.spp=100
        
        #render_objects_by_position(visii_pybullet,obj_base_position)
        render_objects_by_force(visii_pybullet,force_max)
        #set a random light intensity      
        visii.set_dome_light_intensity(random.uniform(1.1,2))

        #rotate the enviroment by a random angle about the z axis 
        #basically we are locking around the enviroment 
        q1=visii.angleAxis(random.uniform(-visii.pi(),visii.pi()),visii.vec3(0,0,1))
        visii.set_dome_light_rotation(q1)    

        print(f"{str(i_render+start_image_number).zfill(5)}/{str(opt.nb_frames).zfill(5)}/{backgrounds[i_background]}")

        #render the actual image
        generate_images(opt.width,
                                opt.height,
                                opt.spp,
                                opt.outf,
                                i_render+start_image_number)
        
        #generate the image annotations
        generate_annotations(opt.width,
                                opt.height,
                                i_render+start_image_number,
                                names_to_export,
                                camera_struct,
                                opt.visibility_fraction,
                                opt.outf)
        
        i_render +=1
            
        if i_render % n_frames_per_background ==0 :
            i_background+=1
        #remove previuos skybox
        if dome_tex.is_initialized():
            visii.texture.remove(skybox_name)
        elif flag==1 and i_background<len(backgrounds):
            #simply change the color if the uniform background already exists
            color=backgrounds[i_background]
            corrected_color=np.array(background_colors[color])/255
            background_material.set_base_color(corrected_color)

visii.deinitialize()
final_time=time.time()
print("skybox",skybox_time-start_time)
print("uniform",final_time-skybox_time)
print("total",final_time-start_time)


