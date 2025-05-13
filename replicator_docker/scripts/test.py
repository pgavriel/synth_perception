print("\n== SCRIPT START ==\n")
import time
start_time = time.time()
import omni.replicator.core as rep
import omni.kit.material.library as lib
import omni.usd
import omni.kit.commands
from pxr import Sdf
import os.path
from os.path import join, dirname, basename
import sys
from math import ceil
import random

# sys.path.append(join(dirname(__file__), '..'))
sys.path.append(dirname(__file__))
import utilities as util
from utilities import clamp
    
def create_camera(config):
    camera_position = tuple(config["position"])
    camera_lookat = tuple(config["look_at"])

    camera = rep.create.camera(
        position=camera_position,
        look_at=camera_lookat,
        focal_length=config["focal_length"],
        focus_distance=config["focus_distance"]
    )
    return camera


def create_lighting(config):
    lookat = (0,0,config["background"]["depth"])
    config = config["light"]
    lights = []
    light_type = config["type"]
    n = config["num_lights"]
    for i in range(n):
        print(f"{i+1} - Creating Light Type: {light_type}")
        # position_x=rep.distribution.uniform(config["xy_range"][0],config["xy_range"][1]),
        # position_y=rep.distribution.uniform(config["xy_range"][0],config["xy_range"][1]),
        # position_z=rep.distribution.uniform(config["z_range"][0],config["z_range"][1]),
        # sphere_light = rep.create.light(
        #     light_type="Sphere",
        #     temperature=rep.distribution.normal(6500, 500),
        #     intensity=rep.distribution.normal(35000, 5000),
        #     position=rep.distribution.uniform((-300, -300, -300), (300, 300, 300)),
        #     scale=rep.distribution.uniform(50, 100),
        #     count=2
        # )
        light = rep.create.light(
            light_type=light_type,
            position=(0,0,-100),
            look_at=lookat)#,
        #     intensity=rep.distribution.normal(35000, 5000),
        #     temperature=rep.distribution.normal(6500, 500)
        # )
        lights.append(light)
    return lights

@rep.randomizer.register
def randomize_lighting(config):
    pass

# @rep.randomizer.register
# def random_background(depth=1000, density=0.05, object_scale_range=(20, 50)):
#     fov = 60  # degrees
#     aspect = 1024/640
#     width, height = util.compute_plane_size(fov, aspect, depth)

#     num_x = int(width * density)
#     num_y = int(height * density)

#     shapes = []
#     for _ in range(num_x * num_y):
#         shape_type = rep.distribution.choice(["cube", "sphere", "cylinder"])
#         # shape = rep.create.primitive(shape_type=shape_type)
#         # shape = rep.create.from_primitive(prim_type="Cube", semantics=[("class", "background")])

#         shape = rep.create.cylinder()
#         if shape_type == 'cube':
#             shape = rep.create.cube()
#         elif shape_type == 'sphere':
#             shape = rep.create.sphere()
#         elif shape_type == 'cylinder':
#             shape = rep.create.cylinder()
        
#         translation = rep.distribution.uniform((-width/2, -height/2, depth), (width/2, height/2, depth))
#         scale = rep.distribution.uniform(object_scale_range[0], object_scale_range[1])
#         rotation = rep.distribution.uniform((-180, -180, -180), (180, 180, 180))
#         # shape.set(scale=(scale, scale, scale))

#         # x = rep.distribution.uniform(-width/2, width/2)
#         # y = rep.distribution.uniform(-height/2, height/2)
#         # shape.set(position=(x, y, depth))

#         # shape.set(rotation=rep.distribution.uniform((-180, -180, -180), (180, 180, 180)))
#         # shape.set(color=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))
#         shape = rep.modify.pose(input_prims=shape,position=translation,rotation=rotation,scale=scale)
#         shapes.append(shape)

#     return shapes

def get_foreground_bound(config):
    aspect = config["output_size"][0]/config["output_size"][1]

    config = config["foreground"]
    depth = config["max_dist"] - config["min_dist"]
    center_z = (config["max_dist"]+config["min_dist"])/2
    position = (0,0,center_z)
    scale = (config["scale_xy"]*aspect,config["scale_xy"],config["scale_z"])
    # size = (config["width"],config["height"],depth)
    size = (config["height"]*aspect,config["height"],depth)
    visible = config["show_bounds"]
    # Define Scatter volume
    sample_volume = rep.create.cube(
        position=position,
        scale=scale,
        visible=config["show_bounds"]
    )
    # with sample_volume:
    #     rep.modify.pose(size=size)
    print(f"Foreground Bound (Cube):\
          \n\tPosition: {position}\
          \n\tScale: {scale}\
          \n\tSize: {size}\
          \n\tVisible: {visible}")

    return sample_volume

def get_background_plane(config):
    aspect = config["output_size"][0]/config["output_size"][1]
    config = config["background"]
    bkg_plane = rep.create.plane(
        position=(0,0,config["depth"]),
        rotation=(90,0,0),
        scale=(config["scale"]*aspect,config["scale"],config["scale"]),
        visible=config["show_plane"]
    )
    return bkg_plane

def create_background_objects(config):
    config = config["background"]
    # Establish objects to spawn and object counts
    if config["object_count"] == 0:
        return rep.create.group([])
    shapes = []
    for obj in config["objects"]:
        if config["objects"][obj] == True: shapes.append(obj)
    print(f"Spawning {len(shapes)} shapes: {shapes}")
    # Currently does not precisely spawn "object_count" objects
    obj_count = ceil(config["object_count"]/len(shapes))
    print(f"Spawning {obj_count} instance of each shape")
    # Establish distribution for scaling
    # s_min = config["object_scale"][1] * config["dim_scale_bounds"][0]
    # s_max = config["object_scale"][1] * config["dim_scale_bounds"][1]
    s_min = config["dim_scale_bounds"][0]
    s_max = config["dim_scale_bounds"][1]
    scale_dist = rep.distribution.uniform((s_min,s_min,s_min),(s_max,s_max,s_max))
    # Initial spawn position
    spawn_pos = (-100,-100,-100)
    # Create object instances
    objects_list = []
    default_mat = rep.create.material_omnipbr(diffuse=(0,230,0))
    if config["objects"]["cone"] == True:
        for i in range(obj_count):
            objects_list.append(rep.create.cone(position=spawn_pos,scale=scale_dist))#,material=default_mat))
    if config["objects"]["cube"] == True:
        for i in range(obj_count):
            objects_list.append(rep.create.cube(position=spawn_pos,scale=scale_dist))#,material=default_mat))
    if config["objects"]["cylinder"] == True:
        for i in range(obj_count):
            objects_list.append(rep.create.cylinder(position=spawn_pos,scale=scale_dist))#,material=default_mat))
    if config["objects"]["sphere"] == True:
        for i in range(obj_count):
            objects_list.append(rep.create.sphere(position=spawn_pos,scale=scale_dist))#,material=default_mat))
    if config["objects"]["torus"] == True:
        for i in range(obj_count):
            objects_list.append(rep.create.torus(position=spawn_pos,scale=scale_dist))#,material=default_mat))

    # Create object group
    background_object_group = rep.create.group(objects_list)
    return background_object_group

def load_diffuse_materials(config, verbose=True):
    """
    Recursively search for *_diff.jpg files in selected subdirectories under root_dir,
    and create an OmniPBR material for each. If a corresponding *_norm.jpg is found,
    use it as the roughness texture. If no *_diff.jpg files are found in a subfolder,
    create materials from all .jpg and .png files in that subfolder.

    Args:
        config (dict): Expects a 'root' directory, and 'include_folders' list of subfolders
        verbose (bool): Whether to print status messages.

    Returns:
        Replicator Group: Group of rep material objects created from texture images.
    """
    materials = []
    root_dir = config["root"]
    subdirs = config["include_folders"]
    n = config["variants"]

    for subdir in subdirs:
        full_path = os.path.join(root_dir, subdir)
        if verbose:
            print(f"\nSearching in: {full_path}")

        diff_textures_found = False
        fallback_textures = []

        for dirpath, _, filenames in os.walk(full_path):
            for file in filenames:
                lower_file = file.lower()

                # Track all jpg/pngs for fallback if needed
                if lower_file.endswith(".jpg") or lower_file.endswith(".png"):
                    fallback_textures.append(os.path.join(dirpath, file))

                if lower_file.endswith("_diff.jpg"):
                    diff_textures_found = True
                    diff_path = os.path.join(dirpath, file)
                    base_name = file[:-9]  # Strip '_diff.jpg'
                    norm_file = base_name + "_norm.jpg"
                    norm_path = os.path.join(dirpath, norm_file)

                    if os.path.exists(norm_path):
                        # mat = rep.create.material_omnipbr(
                        #     diffuse_texture=diff_path,
                        #     roughness_texture=norm_path
                        # )
                        for i in range(n):
                            mat = create_randomized_material(config,diff_path,norm_path)
                            materials.append(mat)
                    else:
                        if verbose:
                            print(f"  No normal texture found for '{base_name}'")
                        # mat = rep.create.material_omnipbr(diffuse_texture=diff_path)
                        for i in range(n):
                            mat = create_randomized_material(config,diff_path)
                            materials.append(mat)

                    # materials.append(mat)

        # If no _diff.jpg textures were found, fallback to all jpg/pngs
        if not diff_textures_found:
            if verbose:
                print(f"  No *_diff.jpg files found — using all .jpg/.png as diffuse")
            for tex_path in fallback_textures:
                # mat = rep.create.material_omnipbr(diffuse_texture=tex_path)
                for i in range(n):
                    mat = create_randomized_material(config,tex_path)
                    materials.append(mat)

    if verbose:
        print(f"Created {n} variants for each material")
        print(f"Total materials created: {len(materials)}")
        # for t in materials:
        #     print(f" > {t}")

    # return materials
    return rep.create.group(materials)

def load_mdl_materials(config, verbose=True):
    """
    Recursively search for *.mdl files in selected subdirectories under root_dir,
    and return the list of materials.

    Args:
        config (dict): Expects a 'root' directory, and 'include_folders' list of subfolders
        verbose (bool): Whether to print status messages.

    Returns:
        list: List of rep material objects created from diffuse textures.
    """
    materials = []
    root_dir = config["root"]
    subdirs = config["include_folders"]

    for subdir in subdirs:
        full_path = os.path.join(root_dir, subdir)
        if verbose:
            print(f"Searching in: {full_path}")
        for dirpath, _, filenames in os.walk(full_path):
            for file in filenames:
                if file.endswith(".mdl"):
                    material_path = os.path.join(dirpath, file)
                    materials.append(material_path)

    if verbose:
        print(f"Total materials created: {len(materials)}")
        for m in materials:
            print(f" > {m}")

    return rep.create.group(materials)
    
def create_randomized_material(config, diff_texture_path, rough_texture_path=None):
    roughness = random.uniform(config["roughness_range"][0], config["roughness_range"][1]) # Should be bounded [0,1]
    metallic = random.uniform(config["metallic_range"][0], config["metallic_range"][1]) # Should be bounded [0,1]
    specular = random.uniform(config["specular_range"][0], config["specular_range"][1]) # Should be bounded [0,1]
    # print(f"R:{roughness:.2f} - M:{metallic:.2f} - S:{specular:.2f}")
    mat = rep.create.material_omnipbr(
        diffuse_texture=diff_texture_path,
        roughness_texture=rough_texture_path,
        roughness=roughness,
        metallic=metallic,
        specular=specular
    )
    # print(f"Mat: {success} {type(result)} - {result}")
    # util.print_object_info(mat,"Material")
    return mat

# Load Configuration json file
config_folder = "/home/ubuntu/config"
config = util.load_json_config(join(config_folder,"config_a.json"))

#Render Settings
rep.settings.set_render_rtx_realtime(antialiasing="FXAA")

model_folder = config["foreground"]["model_root"]
models = rep.utils.get_usd_files(model_folder)
print(f"Models: ({model_folder})")
if len(models) == 0: print("None")
for m in models:
    print(f"> {m}")

# Failed MDL testing
# mdl_file="/home/ubuntu/materials/Ground/Mulch.mdl"
# mat_name="test"
# omni.kit.commands.execute('CreateAndBindMdlMaterialFromLibrary',
#                             mdl_name=mdl_file,
#                             mtl_name=mat_name
# )

# New Scenario ========== ========== ========== ========== ========== ========== ========== ========== 
with rep.new_layer():

    stage = omni.usd.get_context().get_stage()
    print(f"Stage: {stage}")
    # Create Camera
    camera = create_camera(config["camera"])

    # Create Lighting (Create function for different types)
    lc = config["light"]
    lights = create_lighting(config)
    light_color_randomizer = rep.distribution.uniform(
        (lc["r_range"][0],lc["g_range"][0],lc["b_range"][0]),
        (lc["r_range"][1],lc["g_range"][1],lc["b_range"][1])
    )

    # Output render size
    render_product = rep.create.render_product(camera, tuple(config["output_size"]))

    # Create Background
    plane = get_background_plane(config)
    if config["background"]["spawn_objects"] == True:
        background_group = create_background_objects(config)
    background_materials = load_diffuse_materials(config["background"]["materials"])
    # mdl_paths = load_mdl_materials(config["background"]["materials"])

    # Create Foreground objects
    print("Getting Engine...")
    engine_usd = rep.create.from_usd("/home/ubuntu/models/engine4.usd",
                                 semantics=[('class', 'engine')], 
                                 count=4)
    with engine_usd:
        rep.modify.pose(
            position=(0,0,500),
            scale = 1000
        )
    # util.print_object_info(engine_usd,"Engine")

    # Create Foreground Spawning Volume
    # spawn_volume = get_foreground_bound(config)

    # Generate N frames ========== ========== ========== ========== ========== ========== ========== 
    with rep.trigger.on_frame(max_execs=config["num_frames"],rt_subframes=config["num_subframes"]):

        #  RANDOMIZE LIGHT ========== ========== ========== ========== 
        rep.randomizer.randomize_lighting(config)
        with rep.create.group(lights):
            rep.modify.pose(
                position_x=rep.distribution.uniform(lc["xy_range"][0],lc["xy_range"][1]),
                position_y=rep.distribution.uniform(lc["xy_range"][0],lc["xy_range"][1]),
                position_z=rep.distribution.uniform(lc["z_range"][0],lc["z_range"][1]),
                look_at=(0,0,config["background"]["depth"])
            )
            if lc["randomize_color"] == True:
                rep.modify.attribute("color",light_color_randomizer)
            rep.modify.attribute("intensity",rep.distribution.uniform(lc["intensity"][0],lc["intensity"][1]))
            # rep.modify.attribute("exposure",rep.distribution.uniform(lc["exposure"][0],lc["exposure"][1]))
        
        #  RANDOMIZE BACKGROUND PLANE ========== ========== ========== ========== 
        with rep.create.group([plane]):
            # rep.randomizer.color(
            #     colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1))
            # )
            rep.randomizer.materials(background_materials)
        
        # RANDOMIZE BACKGROUND OBJECTS ========== ========== ========== ========== 
        #TODO: Add material/texture randomization
        if config["background"]["spawn_objects"] == True:
            with background_group:
                rep.randomizer.scatter_2d(plane,check_for_collisions=config["background"]["check_collisions"])
                # rep.randomizer.color(
                #     colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1))
                # )
                rep.randomizer.materials(background_materials)
                rep.randomizer.rotation()
                rep.modify.pose(
                    scale = rep.distribution.uniform(config["background"]["object_scale"][0], config["background"]["object_scale"][1])
                )

        # RANDOMIZE FOREGROUND OBJECTS ========== ========== ========== ========== 
        with rep.create.group([engine_usd]):
            # rep.randomizer.scatter_3d(spawn_volume)
            # rep.randomizer.scatter_2d(plane)
            # rep.randomizer.color(
            #     colors=rep.distribution.uniform((0.2, 0.2, 0.2), (1, 1, 1))
            # )
            rep.randomizer.rotation()
            rep.modify.pose(
                position=rep.distribution.uniform((-300, -300, 1000), (300, 300, 1700)))
            #     scale=rep.distribution.uniform(0.1, 0.5))

    # Initialize and attach writer
    writer = rep.WriterRegistry.get("BasicWriter")

    # Create output directory
    output_root = config["output_root"]
    output_dir = util.create_incremental_dir(output_root,config["output_name"])

    # Configure data output types 
    outputs = config["output_types"]
    writer.initialize(
        output_dir=output_dir,
        rgb=outputs["rgb"],
        bounding_box_2d_tight=outputs["bounding_box_2d_tight"],
        bounding_box_2d_loose=outputs["bounding_box_2d_loose"],
        semantic_segmentation=outputs["semantic_segmentation"],
        instance_segmentation=outputs["instance_segmentation"],
        distance_to_camera=outputs["distance_to_camera"],
        distance_to_image_plane=outputs["distance_to_image_plane"],
        bounding_box_3d=outputs["bounding_box_3d"],
        occlusion=outputs["occlusion"],
        normals=outputs["normals"],
    )

    writer.attach([render_product])

    # Run scenario
    rep.orchestrator.run()

    # Script timer
    end_time = time.time()
    total_time = end_time - start_time
    time_per_image = total_time / config["num_frames"] if config["num_frames"] > 0 else float('inf')
    print(f"✅ Total runtime: {total_time:.2f} seconds")
    print(f"🕒 Average time per image: {time_per_image:.4f} seconds")

    print(f"Completed Scenario: {basename(output_dir)}")
    print("\n== DONE ==\n")