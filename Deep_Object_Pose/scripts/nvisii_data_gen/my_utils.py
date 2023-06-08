import nvisii as visii
import glob
import random
from utils import *
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation


def create_camera(world_bullet_to_world_nvisii,focal_length=None,fx=None,fy=None,cx=None,cy=None, width=640,height=480,fov=0.785398,debug=False):
    if focal_length:
        camera = visii.entity.create(
        name = "camera",
        transform = visii.transform.create("camera"),
        camera = visii.camera.create_from_intrinsics(
            name = "camera",
            fx=focal_length,
            fy=focal_length,
            cx=(width / 2),
            cy=(height / 2),
            width=width,
            height=height
        )
    )
        
    elif fx and fy and cx and cy:
        camera = visii.entity.create(
        name = "camera",
        transform = visii.transform.create("camera"),
        camera = visii.camera.create_from_intrinsics(
            name = "camera",
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height
        )
    )
    else:
        camera = visii.entity.create(
        name = "camera",
        transform = visii.transform.create("camera"),
        camera = visii.camera.create_perspective_from_fov(
            name = "camera",
            field_of_view =fov, #45 degree 
            aspect = float(width)/float(height)
        )
    )
    #world_bullet_to_world_nvisii is the matrix from world bullet frame to nvisii bullet frame
    #set the position of the nvisii world frame in bullet
    bullet_camera_pos=world_bullet_to_world_nvisii[0:3,3]

    #make the camera a phisycal object in this case a sphere
    p.createMultiBody(
        baseCollisionShapeIndex = p.createCollisionShape(p.GEOM_SPHERE,0.05),
        baseVisualShapeIndex = p.createVisualShape(p.GEOM_SPHERE,0.05),
        basePosition =bullet_camera_pos,
        baseOrientation=Rotation.from_matrix(world_bullet_to_world_nvisii[0:3,0:3]).as_quat()
    )

    #draw the camera frame in bullet if required
    if debug:
        p.addUserDebugLine(lineFromXYZ = bullet_camera_pos,lineToXYZ=bullet_camera_pos+world_bullet_to_world_nvisii[0:3,0],lineColorRGB=(1,0,0),lifeTime=0,lineWidth=2)
        p.addUserDebugLine(lineFromXYZ = bullet_camera_pos,lineToXYZ=bullet_camera_pos+world_bullet_to_world_nvisii[0:3,1],lineColorRGB=(0,1,0),lifeTime=0,lineWidth=2)
        p.addUserDebugLine(lineFromXYZ = bullet_camera_pos,lineToXYZ=bullet_camera_pos+world_bullet_to_world_nvisii[0:3,2],lineColorRGB=(0,0,1),lifeTime=0,lineWidth=2)           

    return camera

def set_skybox(single_skybox_path=None,skybox_folder="dome_hdri_haven",skybox_name='dome_text'):
    # lets turn off the ambiant lights
    # load a random skybox from the folder
    #load one skybox
    if single_skybox_path:
        skybox_selection =single_skybox_path

    else:
    #or chose one at random
        skyboxes = glob.glob(f'{skybox_folder}/*.hdr')
        skybox_selection = skyboxes[random.randint(0,len(skyboxes)-1)]
    
    dome_tex = visii.texture.create_from_file(skybox_name,skybox_selection)
    visii.set_dome_light_texture(dome_tex)
    return dome_tex

def set_uniform_background(color,max_distance):
    
    background = visii.entity.create(
            name = "background",
            mesh = visii.mesh.create_plane("mesh_background"),
            transform = visii.transform.create("transform_background"),
            material = visii.material.create("material_background")
        )
    

    mat = background.get_material()
    mat.set_roughness(0.7)
    mat.set_base_color(color)
    # Make the wall large and far
    #ATTENTION: since the wall is rendered as an object the depth will count it as an object like the others, depth associated with 
    #the wall MUST be removed for training
    trans = background.get_transform()
    trans.set_scale((10,10,10))
    #place the colored plane perpendicolar the view axis (x axis)
    trans.set_position(visii.vec3(max_distance,0,0))   
    trans.set_rotation(visii.angleAxis(visii.pi()/2,visii.vec3(0,1,0))) 

    return mat


def adding_mesh_object(
        name, 
        obj_to_load, 
        texture_to_load, 
        obj_base_position,
        model_info_path=None, 
        scale=1, 
        debug=False
    ):

    mesh_loaded = {}
   
    if texture_to_load is None:
        toys = load_obj_scene(obj_to_load)
        if len(toys) > 1: 
            print("more than one model in the object, \
                   materials might be wrong!")
        toy_transform = visii.entity.get(toys[0]).get_transform()
        toy_material = visii.entity.get(toys[0]).get_material()
        toy_mesh = visii.entity.get(toys[0]).get_mesh()        

        obj_export = visii.entity.create(
                name = name,
                transform = visii.transform.create(
                    name = name, 
                    position = toy_transform.get_position(),
                    rotation = toy_transform.get_rotation(),
                    scale = toy_transform.get_scale(),
                ),
                material = toy_material,
                mesh = visii.mesh.create_from_file(name,obj_to_load),
            )

        toy_transform = obj_export.get_transform()
        obj_export.get_material().set_roughness(random.uniform(0.1, 0.5))

        for toy in toys:
            visii.entity.remove(toy)

        toys = [name]
    else:
        toys = [name]

        if obj_to_load in mesh_loaded:
            toy_mesh = mesh_loaded[obj_to_load]
        else:
            toy_mesh = visii.mesh.create_from_file(name, obj_to_load)
            mesh_loaded[obj_to_load] = toy_mesh

        toy = visii.entity.create(
            name=name,
            transform=visii.transform.create(name),
            mesh=toy_mesh,
            material=visii.material.create(name)
        )

        toy_rgb_tex = visii.texture.create_from_file(name, texture_to_load)
        toy.get_material().set_base_color_texture(toy_rgb_tex)
        toy.get_material().set_roughness(random.uniform(0.1, 0.5))

        toy_transform = toy.get_transform()

    ###########################

    toy_transform.set_scale(visii.vec3(scale))
    toy_transform.set_position(visii.vec3(obj_base_position))

    quat=visii.quat(random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1))
    toy_transform.set_rotation(quat)

    # add symmetry_corrected transform
    child_transform = visii.transform.create(f"{toy_transform.get_name()}_symmetry_corrected")

    child_transform.set_parent(toy_transform)

    # store symmetry transforms for later use.
    symmetry_transforms = get_symmetry_transformations(model_info_path)
    id_pybullet = create_physics(name, mass=(np.random.rand() * 5))
    visii_pybullet_element={
            'visii_id': name,
            'bullet_id': id_pybullet,
            'base_rot': None,
            'model_info': {},
            'symmetry_transforms': symmetry_transforms
    }
    
    for entity_name in toys:
        add_cuboid(entity_name, scale=scale, debug=debug)

    return visii_pybullet_element,toys

#mesh_path and texture_path are fixed due to the structure of their models  
def load_objects(parameters,obj_base_position,debug=False):
    visii_pybullet=[] 
    names_to_export = []

    for i_obj in range(parameters['n_objects']):
        objects_folder =parameters['folder']
        scale=parameters['scale']
        toy_to_load = objects_folder[random.randint(0, len(objects_folder) - 1)]
        obj_to_load = toy_to_load + parameters['mesh_path']
        texture_to_load = toy_to_load + parameters['textures_path']
        name =  parameters['name_prefix']+ toy_to_load.split('/')[-2] + f"_{i_obj}"
        visii_pybullet_element,names=adding_mesh_object(name, obj_to_load, texture_to_load,obj_base_position,scale=scale, debug=debug)
        visii_pybullet.append(visii_pybullet_element)
        names_to_export.extend(names)

    return visii_pybullet,names_to_export

def render_objects_by_position(objects,obj_base_position):
    for entry in objects:
        visii.entity.get(entry['visii_id']).get_transform().set_position(obj_base_position)
        #set a random orientation
        quat=visii.quat(random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0,1))
        visii.entity.get(entry['visii_id']).get_transform().set_rotation(quat)


def render_objects_by_force(objects,force_max):
    object_position=0.01
    for entry in objects:
        visii.entity.get(entry['visii_id']).get_transform().set_position(
            visii.entity.get(entry['visii_id']).get_transform().get_position(),
            previous = True
        )
        visii.entity.get(entry['visii_id']).get_transform().set_rotation(
            visii.entity.get(entry['visii_id']).get_transform().get_rotation(),
            previous = True
        )
      
        update_pose(entry)

        p.applyExternalForce(
                entry['bullet_id'],
                -1,
                [   random.uniform(-force_max[0],force_max[0]),
                    random.uniform(-force_max[1],force_max[1]),
                    random.uniform(-force_max[2],force_max[2])],
                [  random.uniform(-object_position,object_position),
                    random.uniform(-object_position,object_position),
                    random.uniform(-object_position,object_position)],
                flags=p.WORLD_FRAME
            )
        
"""def render_with_camera_movement(r,phi,theta,camera,obj_base_position):
    pos=(-r*np.cos(phi)*np.sin(theta),r*np.sin(phi),-r*np.cos(phi)*np.cos(theta))
    Tc_w=np.zeros((3,3))
    #transponse to write column-wise 
    Tc_w[:3,:3]=Rotation.from_euler('zy',[theta,-phi]).as_matrix()
    Tc_w[0:3,1]=(-1)*Tc_w[0:3,1]
    Tc_w[0:3,2]=(-1)*Tc_w[0:3,2]
    camera.get_transform().look_at(at=obj_base_position,eye=pos,up=Tc_w[0:3,1])
    eye,up=pos,Tc_w[0:3,1]
    return eye,up"""


def generate_images(width,height,spp,outf,i_render):
    visii.sample_pixel_area(
            x_sample_interval = (0,1),
            y_sample_interval = (0,1))
    
    visii.render_to_file(
            width=int(width),
            height=int(height),
            samples_per_pixel=int(spp),
            file_path=f"{outf}/{str(i_render).zfill(5)}.png"
        )
    

    visii.sample_pixel_area(
        x_sample_interval = (.5,.5),
        y_sample_interval = (.5,.5))
    
    visii.sample_time_interval((1,1))

    visii.render_data_to_file(
            width=width,
            height=height,
            start_frame=0,
            frame_count=1,
            bounce=int(0),
            options="entity_id",
            file_path = f"{outf}/{str(i_render).zfill(5)}.seg.exr"
        )

    visii.render_data_to_file(
            width=width,
            height=height,
            start_frame=0,
            frame_count=1,
            bounce=int(0),
            options="depth",
            file_path = f"{outf}/{str(i_render).zfill(5)}.depth.exr"
        )
    
def generate_annotations(width,height,i_render,names_to_export,random_camera_movement,visibility_fraction,outf):
        segmentation_mask = visii.render_data(
            width=int(width),
            height=int(height),
            start_frame=0,
            frame_count=1,
            bounce=int(0),
            options="entity_id",
        )
    
        segmentation_mask = np.array(segmentation_mask).reshape((height, width, 4))[:, :, 0]
        
        export_to_ndds_file(
            f"{outf}/{str(i_render).zfill(5)}.json",
            obj_names = names_to_export,
            width = width,
            height = height,
            camera_name = 'camera',
            camera_struct = random_camera_movement,
            segmentation_mask=segmentation_mask,
            compute_visibility_fraction=visibility_fraction,
        )

def create_limiting_plane(position,orientation):
            
    box_col = p.createCollisionShape(p.GEOM_BOX,halfExtents=[2,2,0.1])
    box_vis = p.createVisualShape(p.GEOM_BOX,halfExtents=[2,2,0.1])

    p.createMultiBody(
        baseCollisionShapeIndex = box_col,
        baseVisualShapeIndex=box_vis,
        basePosition = list(position),
        #make the plane perdicolar to the x asis
        baseOrientation=orientation,
        baseMass=0
        )
    
        


