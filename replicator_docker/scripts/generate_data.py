import time
import omni.replicator.core as rep
import os.path
from os.path import join, dirname, basename
import sys
import math
from math import ceil
import random

sys.path.append(dirname(__file__))
import utilities as util
from utilities import clamp
    
def create_camera(config):
    camera_position = tuple(config["position"])
    camera_lookat = tuple(config["look_at"])

    camera = rep.create.camera(
        name="MainCamera",
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
    print(f"\nCreating Lights...")
    for i in range(n):
        print(f"[{i+1}/{n}] Creating Light Type: {light_type}")
        light = rep.create.light(
            light_type=light_type,
            position=(0,0,-100),
            look_at=lookat)

        lights.append(light)
    return lights


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

    # Establish distribution for axis scaling
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
                        for i in range(n):
                            mat = create_randomized_material(config,diff_path,norm_path)
                            materials.append(mat)
                    else:
                        if verbose:
                            print(f"  No normal texture found for '{base_name}'")
                        for i in range(n):
                            mat = create_randomized_material(config,diff_path)
                            materials.append(mat)

        # If no _diff.jpg textures were found, fallback to all jpg/pngs
        if not diff_textures_found:
            if verbose:
                print(f"  No *_diff.jpg files found — using all .jpg/.png as diffuse")
            for tex_path in fallback_textures:
                for i in range(n):
                    mat = create_randomized_material(config,tex_path)
                    materials.append(mat)

    if verbose:
        print(f"Created {n} variants for each material")
        print(f"Total materials created: {len(materials)}")
        # for t in materials:
        #     print(f" > {t}")

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

def create_foreground_objects(config):
    # Expects a dictionary of objects of the form "USD File: Class Label"
    aspect = config["output_size"][0]/config["output_size"][1]
    objects = []
    config = config["foreground"]
    n = config["object_count"]
    obj = config["objects"]
    print(f"\nCreating {n} foreground objects...")
    for i in range(n):
        choice = random.sample(list(config["objects"].keys()),1)[0]
        model_file = join(config["model_root"],choice)
        class_name = obj[choice]
        print(f"[{i+1}/{n}] Model: {model_file} - Class: {class_name}")
        if os.path.isfile(model_file):
            # Load Model File
            usd_model = rep.create.from_usd(model_file,
                                    semantics=[('class', class_name)], 
                                    count=1)
            with usd_model:
                rep.modify.pose(
                    position=tuple(config["default_pos"]),
                    rotation=tuple(config["default_rot"]),
                    scale = config["default_scale"]
                )

            objects.append(usd_model)
        elif choice == "Plane":
            plane = rep.create.plane(
                position=tuple(config["default_pos"]),
                rotation=tuple(config["default_rot"]),
                scale=(config["default_scale"]*aspect,config["default_scale"],config["default_scale"])
            )
            objects.append(plane)
        elif choice == "Sphere":
            plane = rep.create.sphere(
                position=tuple(config["default_pos"]),
                rotation=tuple(config["default_rot"]),
                scale=(config["default_scale"])
            )
            objects.append(plane)
        else:
            print(f"ERROR: Failed to find model")
            model_folder = config["model_root"]
            models = rep.utils.get_usd_files(model_folder)
            print(f"Models in model_root: ({model_folder})")
            if len(models) == 0: print("None")
            for m in models:
                print(f"> {m}")
            return None
    return objects

def get_random_foreground_position(config,mod_size=True):
    aspect_ratio = config["output_size"][0]/config["output_size"][1]
    fc = config["foreground"]
    vfov_deg = fc["spawn_vfov"]
    z_pos = random.uniform(fc["z_range"][0],fc["z_range"][1])
    # z_pos = rep.distribution.uniform(fc["z_range"][0],fc["z_range"][1]).get_output()
    # util.print_object_info(z_pos)

    vfov_rad = math.radians(vfov_deg)
    
    # Height at depth z
    y_bound = math.tan(vfov_rad / 2) * z_pos
    # y_pos = random.uniform(-y_bound,y_bound)

    # Width = height * aspect
    x_bound = y_bound * aspect_ratio
    # x_pos = random.uniform(-x_bound,x_bound)

    position = rep.distribution.uniform((-x_bound,-y_bound,z_pos),(x_bound,y_bound,z_pos))
    size = (2*x_bound,2*y_bound,2*y_bound)
    # print(f"POS: {position} - SIZE BOUNDS: {size}")

    return rep.modify.pose(position=position)
    # return position
rep.randomizer.register(get_random_foreground_position)

# default_config_dir = "/home/ubuntu/config"
# default_config = "config_a.json"
# config_path = join(default_config_dir,default_config)
def run_data_generation_scenario(config_path):
    # MAIN START ========== ========== ========== ========== ========== ========== ========== ========== 
    print("\n== SCENARIO START ==\n")
    start_time = time.time()

    # Load Configuration json file
    print(f"Loading Config: {config_path}")
    config = util.load_json_config(config_path)
    print(f"Scenario: {config['scenario_name']}")

    # Render Settings
    rep.settings.set_render_rtx_realtime(antialiasing="FXAA")

    # Failed MDL testing
    # mdl_file="/home/ubuntu/materials/Ground/Mulch.mdl"
    # mat_name="test"
    # omni.kit.commands.execute('CreateAndBindMdlMaterialFromLibrary',
    #                             mdl_name=mdl_file,
    #                             mtl_name=mat_name
    # )

    # New Scenario ========== ========== ========== ========== ========== ========== ========== ========== 
    with rep.new_layer():

        # Create Camera
        camera = create_camera(config["camera"])
        camera_prim = rep.get.prims(path_pattern="(.*?)") 
        util.print_object_info(camera._get_prims())
        print(camera._get_prims())

        # Output render size
        render_product = rep.create.render_product(camera, tuple(config["output_size"]))

        # Create Lighting (Create function for different types)
        lights = create_lighting(config)
        lc = config["light"]
        light_color_randomizer = rep.distribution.uniform(
            (lc["r_range"][0],lc["g_range"][0],lc["b_range"][0]),
            (lc["r_range"][1],lc["g_range"][1],lc["b_range"][1])
        )

        # Create Background
        #TODO: Define separate plane depth and object depth, just create 2 identical planes
        plane = get_background_plane(config)
        if config["background"]["spawn_objects"] == True:
            background_group = create_background_objects(config)
        background_materials = load_diffuse_materials(config["background"]["materials"])
        # mdl_paths = load_mdl_materials(config["background"]["materials"])

        # Create Foreground objects
        foreground_objects = create_foreground_objects(config)
        fc = config["foreground"]
        
        # Generate N frames ========== ========== ========== ========== ========== ========== ========== 
        with rep.trigger.on_frame(max_execs=config["num_frames"],rt_subframes=config["num_subframes"]):

            #  RANDOMIZE LIGHT ========== ========== ========== ========== 
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
            # with foreground_objects:
                # Scatter was not working with loaded USD models
                # rep.randomizer.scatter_3d(spawn_volume)
                # rep.randomizer.scatter_2d(plane)
                # position,size = get_random_foreground_position(config)
                # if fc["random_color"]:
                #     rep.randomizer.color(
                #         colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1))
                #     )
                # Randomize Pose and Rotation
                # if fc["random_rotation"]:
                #     rep.randomizer.rotation()
                # rep.modify.pose(
                #     # position=rep.distribution.uniform((-300, -300, 1000), (300, 300, 1700)))
                #     position=get_random_foreground_position(config)     
                # )
                # rep.randomizer.get_random_foreground_position(config)
                #TODO: Look into model attributes which can be randomized
            for obj in foreground_objects:
                with obj:
                    if fc["random_color"]:
                        rep.randomizer.color(
                            colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1))
                        )
                    if fc["random_rotation"]:
                        rep.randomizer.rotation()
                    rep.randomizer.get_random_foreground_position(config)

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
            pointcloud=outputs["pointclouds"]
        )

        writer.attach([render_product])

        # Run scenario
        rep.orchestrator.run()
        # rep.orchestrator.run_async()



        # Script timer
        end_time = time.time()
        total_time = end_time - start_time
        time_per_image = total_time / config["num_frames"] if config["num_frames"] > 0 else float('inf')
        print(f"\n🕒 Total runtime: {total_time:.2f} seconds")
        print(f"🕒 Average time per image: {time_per_image:.4f} seconds")
        
        # Copy config to data output directory
        if config["copy_config_to_output"]:
            util.save_json(join(output_dir,"config.json"),config)
            print("🖨️ Config copied to output directory.")

        print(f"✅ Completed Scenario: {basename(output_dir)}")
        print("\n== DONE ==\n")



# Set up default configuration
default_config_dir = "/home/ubuntu/config"
default_config = "dev_config.json"
config = join(default_config_dir,default_config)
print(f"ARGV: {sys.argv}")

if len(sys.argv) > 6:
    config_list = sys.argv[6:]
    # Ensure .json suffix so it may be excluded in cmd line arguments
    config_list = [name if name.lower().endswith('.json') else name + '.json' for name in config_list]
    print(f"Config List: {config_list}")
    for conf in config_list:
        print(f"Running scenario for config: {conf}")
        config = join(default_config_dir,conf)
        run_data_generation_scenario(config)
else:
    print(f"WARNING: No config file arguments provided")
    print(f" > Using default config \'{default_config}\'")
    run_data_generation_scenario(config)
