import time
import omni.replicator.core as rep
import omni.usd
import os.path
from os.path import join, dirname, basename
import sys
import math
from math import ceil
import random
# from typing import Optional
# from pxr import Sdf, Usd

sys.path.append(dirname(__file__))
import utilities as util
from utilities import clamp
from utilities import print_prim, inspect_object, get_all_prim_paths
    

def create_camera(config):
    print(f"\nCreating Camera...")
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
    print(f"\nCreating Background Plane...")
    aspect = config["output_size"][0]/config["output_size"][1]
    config = config["background"]
    bkg_plane = rep.create.plane(
        position=(0,0,config["depth"]),
        rotation=(90,0,0),
        scale=(config["scale"]*aspect,config["scale"],config["scale"]),
        visible=config["show_plane"]
    )
    return bkg_plane

def get_background_object_plane(config):
    print(f"\nCreating Background Object Plane...")
    aspect = config["output_size"][0]/config["output_size"][1]
    config = config["background"]
    bkg_plane = rep.create.plane(
        position=(0,0,config["depth"]+config["object_plane_offset"]),
        rotation=(90,0,0),
        scale=(config["scale"]*aspect,config["scale"],config["scale"]),
        visible=False
    )
    return bkg_plane

def create_background_objects(config):
    print(f"\nCreating Background Objects...")
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
            cube = rep.create.cube(position=spawn_pos,scale=scale_dist)
            # inspect_object(cube)
            objects_list.append(cube)
            # print_prim(cube.outputs)
            # objects_list.append(rep.create.cube(position=spawn_pos,scale=scale_dist))#,material=default_mat))
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
    print(f"\nCreating Materials...")
    materials = []
    root_dir = config["root"]
    subdirs = config["include_folders"]
    n = config["variants"]

    for subdir in subdirs:
        full_path = os.path.join(root_dir, subdir)
        if verbose:
            print(f"Searching in: {full_path}")

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
    # Expects a dictionary of objects of the form "USD File": "Class Label"
    aspect = config["output_size"][0]/config["output_size"][1]
    objects = []
    config = config["foreground"]
    # n = config["object_count"]
    obj = config["objects"]
    num_instances = config["instance_per_object"]
    usds = []
    # print(f"\nCreating {n} foreground objects...")
    # for i in range(n):
    #     choice = random.sample(list(config["objects"].keys()),1)[0]
    # TODO: Add parameter to create n duplicates of each object
    n = len(list(obj.keys())) * num_instances
    print(f"\nCreating {n} foreground objects...")
    i = 0
    for choice in obj:
        model_file = join(config["model_root"],choice)
        class_name = obj[choice]
        print(f"[{i+1}/{n}] Model: {model_file} - Class: {class_name}")
        i += 1
        if os.path.isfile(model_file):
            # Load Model File
            for x in range(num_instances):
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
                usds.append(model_file)
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
    return objects, usds

def get_random_foreground_position(config):
    """
    Uses the output aspect ratio, and a specified field of view to obtain a random position inside
    the camera frustrum, bounded by the config/foreground/z_range depth range.
    NOTE: Assumes the camera is at (0,0,0) looking in the +Z direction.
    """
    aspect_ratio = config["output_size"][0]/config["output_size"][1]
    fc = config["foreground"]
    # Parameters
    min_depth = fc["z_range"][0]
    max_depth = fc["z_range"][1]
    vfov_deg = fc["spawn_vfov"]  # vertical field of view

    # IMPLEMENTATION 1: ANGULAR RAYS
    # vfov_rad = math.radians(vfov_deg)
    # hfov_rad = 2 * math.atan(math.tan(vfov_rad / 2) * aspect_ratio)

    # depth = random.uniform(min_depth, max_depth)
    # theta = random.uniform(-hfov_rad / 2, hfov_rad / 2)
    # phi = random.uniform(-vfov_rad / 2, vfov_rad / 2)

    # x = depth * math.tan(theta)
    # y = depth * math.tan(phi)
    # z = depth

    # IMPLEMENTATION 2: DEPTH PLANES (SIMPLER)
    # Select a random depth value within the specified range
    z = random.uniform(min_depth,max_depth)
    # Determine the proper XY bounds at the chosen depth
    vfov_rad = math.radians(vfov_deg)
    # # Height at depth z
    y_bound = math.tan(vfov_rad / 2) * z
    # # Width = height * aspect
    x_bound = y_bound * aspect_ratio
    # Randomly select the XY coordinates within the calculated bounds
    x = random.uniform(-x_bound,x_bound)
    y = random.uniform(-y_bound,y_bound)
    
    # Return the chosen position tuple
    # print(f"X:{x:.1f} - Y:{y:.1f} - Z:{z:.1f}")
    return (x, y, z)

# rep.randomizer.register(get_random_foreground_position)

def generate_foreground_positions(config,n=100):
    print(f"Generating {n} possible foreground positions...")
    positions = []
    for i in range(n):
        positions.append(get_random_foreground_position(config))
    print("Done.")
    return positions

def generate_visibility_patterns(fg_objects,config,n=100):
    print(f"Generating {n} possible visibility masks...")
    visible_objects = []
    num_objects = len(fg_objects)
    num_visible = config["visible_object_count"]
    print("num objects: ",num_objects)
    visible_objects = []
    for _ in range(n):
        mask = [False] * num_objects
        true_indices = random.sample(range(num_objects), num_visible)  # pick x unique indices
        for idx in true_indices:
            mask[idx] = True
        visible_objects.append(mask)
    return visible_objects

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

    # New Scenario ========== ========== ========== ========== ========== ========== ========== ========== 
    with rep.new_layer():

        # Create Camera
        camera = create_camera(config["camera"])
        # camera_prim = rep.get.prims(path_pattern="(.*?)") 
        # util.print_object_info(camera._get_prims())
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
        bkg_plane = get_background_plane(config)
        bc = config["background"]
        par = bc["plane_angle_range"]
        if config["background"]["spawn_objects"] == True:
            object_plane = get_background_object_plane(config)
            background_group = create_background_objects(config)
            print("Done")
            
        background_materials = load_diffuse_materials(config["background"]["materials"])
        # mdl_paths = load_mdl_materials(config["background"]["materials"])

        # Create Foreground objects
        fc = config["foreground"]
        if fc["spawn_objects"]:
            foreground_objects, usds = create_foreground_objects(config)
            if fc["random_materials"]:
                foreground_materials = load_diffuse_materials(config["foreground"]["materials"])
            else:
                foreground_materials = None
            n = 2 * config["num_frames"] * fc["visible_object_count"]
            # TODO: Maybe pre-generate a list for each object, one position per frame
            foreground_positions = generate_foreground_positions(config,n)
            # foreground_visible = generate_visibility_patterns(foreground_objects,fc,n)
            pv = min(1.0, fc["visible_object_count"]/len(foreground_objects))
            print(f"Foreground Visibility: PV(True): {pv}, 1-PV(False): {1-pv}")
            # foreground_visibility = rep.distribution.choice([True,False],weights=[pv,1.0-pv])
        
        setup_end_time = time.time()
        total_time = setup_end_time - start_time
        print(f"\n🕒 Total Setup Time: {total_time:.2f} seconds")
        # Generate N frames ========== ========== ========== ========== ========== ========== ========== 
        with rep.trigger.on_frame(max_execs=config["num_frames"],rt_subframes=config["num_subframes"]):
            print("Generating...")
            # RANDOMIZE LIGHT ========== ========== ========== ========== 
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
            
            # RANDOMIZE BACKGROUND PLANE ========== ========== ========== ========== 
            with rep.create.group([bkg_plane]):
                rep.modify.pose(
                    rotation=rep.distribution.uniform((90+par[0],par[0], 0), (90+par[1], par[1], 0)),
                )
                # rep.randomizer.color(
                #     colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1))
                # )
                rep.randomizer.materials(background_materials)
            
            # RANDOMIZE BACKGROUND OBJECTS ========== ========== ========== ========== 
            if config["background"]["spawn_objects"] == True:
                with background_group:
                    rep.randomizer.scatter_2d(object_plane,check_for_collisions=config["background"]["check_collisions"])
                    # rep.randomizer.color(
                    #     colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1))
                    # )
                    rep.randomizer.materials(background_materials)
                    rep.randomizer.rotation()
                    rep.modify.pose(
                        scale = rep.distribution.uniform(config["background"]["object_scale"][0], config["background"]["object_scale"][1])
                    )

            # RANDOMIZE FOREGROUND OBJECTS ========== ========== ========== ========== 
            # TODO: Look into model attributes which can be randomized
            if fc["spawn_objects"]:
                for obj in foreground_objects:
                    with obj:
                        if fc["random_color"]:
                            rep.randomizer.color(
                                colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1))
                            )
                        if fc["random_materials"]:
                            # print("YES RANDOMIZE FG MATERIALS")
                            rep.randomizer.materials(foreground_materials)
                        if fc["random_rotation"]:
                            rep.randomizer.rotation()

                        # rand_pos = get_random_foreground_position(config)
                        if fc["random_position"]:
                            rep.modify.pose(position=rep.distribution.choice(foreground_positions))
                        # rep.randomizer.get_random_foreground_position(config)

                        # Set the probability of being visible as proportional to the desired number of objects to the total
                        # NOTE: This approach should keep "object_count" objects visible ON AVERAGE, but is not strict.
                        rep.modify.visibility(rep.distribution.choice([True,False],weights=[pv,1.0-pv]))

        # Initialize and attach writer
        writer = rep.WriterRegistry.get("BasicWriter")

        # Create output directory
        output_root = config["output_root"]
        output_dir = util.create_incremental_dir(output_root,config["output_name"])

        # Configure data output types 
        if "output_types" in config:
            outputs = config["output_types"]
        else:
            outputs = dict()
            print("WARNING: \"output_types\" not found in config, using defaults...")
        writer.initialize(
            output_dir=output_dir,
            rgb=outputs.get("rgb",True),
            bounding_box_2d_tight=outputs.get("bounding_box_2d_tight",False),
            bounding_box_2d_loose=outputs.get("bounding_box_2d_loose",False),
            semantic_segmentation=outputs.get("semantic_segmentation",False),
            instance_segmentation=outputs.get("instance_segmentation",False),
            distance_to_camera=outputs.get("distance_to_camera",False),
            distance_to_image_plane=outputs.get("distance_to_image_plane",False),
            bounding_box_3d=outputs.get("bounding_box_3d",False),
            occlusion=outputs.get("occlusion",False),
            normals=outputs.get("normals",False),
            pointcloud=outputs.get("pointclouds",False),
            camera_params=outputs.get("camera_params",False)
        )

        writer.attach([render_product])

        # Run scenario
        rep.orchestrator.run()

        # Script timer
        end_time = time.time()
        total_time = end_time - start_time
        generation_time = end_time - setup_end_time
        time_per_image = generation_time / config["num_frames"] if config["num_frames"] > 0 else float('inf')
        print(f"\n🕒 Total runtime: {total_time:.2f} seconds")
        print(f"\n🕒 Total Generation Time: {generation_time:.2f} seconds")
        print(f"🕒 Average time per image: {time_per_image:.4f} seconds")
        
        # Copy config to data output directory
        if config["copy_config_to_output"]:
            util.save_json(join(output_dir,f"config_{basename(output_dir)}.json"),config)
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
