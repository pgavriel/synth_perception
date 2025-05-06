print("\n== SCRIPT START ==\n")

import omni.replicator.core as rep

import os.path
from os.path import join, dirname, basename
import sys
from math import ceil

# sys.path.append(join(dirname(__file__), '..'))
sys.path.append(dirname(__file__))
import utilities as util

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
    light_type = config["type"]
    print(f"Creating Light Type: {light_type}")
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
        position=tuple(config["position"]),
        look_at=(0,0,2000))#,
    #     intensity=rep.distribution.normal(35000, 5000),
    #     temperature=rep.distribution.normal(6500, 500)
    # )
    return light

@rep.randomizer.register
def random_background(depth=1000, density=0.05, object_scale_range=(20, 50)):
    fov = 60  # degrees
    aspect = 1024/640
    width, height = util.compute_plane_size(fov, aspect, depth)

    num_x = int(width * density)
    num_y = int(height * density)

    shapes = []
    for _ in range(num_x * num_y):
        shape_type = rep.distribution.choice(["cube", "sphere", "cylinder"])
        # shape = rep.create.primitive(shape_type=shape_type)
        # shape = rep.create.from_primitive(prim_type="Cube", semantics=[("class", "background")])

        shape = rep.create.cylinder()
        if shape_type == 'cube':
            shape = rep.create.cube()
        elif shape_type == 'sphere':
            shape = rep.create.sphere()
        elif shape_type == 'cylinder':
            shape = rep.create.cylinder()
        
        translation = rep.distribution.uniform((-width/2, -height/2, depth), (width/2, height/2, depth))
        scale = rep.distribution.uniform(object_scale_range[0], object_scale_range[1])
        rotation = rep.distribution.uniform((-180, -180, -180), (180, 180, 180))
        # shape.set(scale=(scale, scale, scale))

        # x = rep.distribution.uniform(-width/2, width/2)
        # y = rep.distribution.uniform(-height/2, height/2)
        # shape.set(position=(x, y, depth))

        # shape.set(rotation=rep.distribution.uniform((-180, -180, -180), (180, 180, 180)))
        # shape.set(color=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))
        shape = rep.modify.pose(input_prims=shape,position=translation,rotation=rotation,scale=scale)
        shapes.append(shape)

    return shapes

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
    s_min = config["object_scale"] * config["dim_scale_bounds"][0]
    s_max = config["object_scale"] * config["dim_scale_bounds"][1]
    scale_dist = rep.distribution.uniform((s_min,s_min,s_min),(s_max,s_max,s_max))
    # Initial spawn position
    spawn_pos = (-100,-100,-100)
    # Create object instances
    objects_list = []
    if config["objects"]["cone"] == True:
        for i in range(obj_count):
            objects_list.append(rep.create.cone(position=spawn_pos,scale=scale_dist))
    if config["objects"]["cube"] == True:
        for i in range(obj_count):
            objects_list.append(rep.create.cube(position=spawn_pos,scale=scale_dist))
    if config["objects"]["cylinder"] == True:
        for i in range(obj_count):
            objects_list.append(rep.create.cylinder(position=spawn_pos,scale=scale_dist))
    if config["objects"]["sphere"] == True:
        for i in range(obj_count):
            objects_list.append(rep.create.sphere(position=spawn_pos,scale=scale_dist))
    if config["objects"]["torus"] == True:
        for i in range(obj_count):
            objects_list.append(rep.create.torus(position=spawn_pos,scale=scale_dist))

    # Create object group
    background_object_group = rep.create.group(objects_list)
    return background_object_group

# Load Configuration json file
config_folder = "/home/ubuntu/config"
config = util.load_json_config(join(config_folder,"config_a.json"))

model_folder = config["model_root"]
models = rep.utils.get_usd_files(model_folder)
print(f"Models: ({model_folder})")
if len(models) == 0: print("None")
for m in models:
    print(f"> {m}")

# New Scenario
with rep.new_layer():

    # Create Camera
    camera = create_camera(config["camera"])

    # Create Lighting (Create function for different types)
    light = create_lighting(config["light"])

    # Output render size
    render_product = rep.create.render_product(camera, tuple(config["output_size"]))

    # Create Background
    pass
    plane = get_background_plane(config)
    background_group = create_background_objects(config)
    # Create Foreground objects
    torus = rep.create.torus(semantics=[('class', 'torus')] , position=(0, -200 , 100),count=3)

    sphere = rep.create.sphere(semantics=[('class', 'sphere')], position=(0, 100, 100))

    cube = rep.create.cube(semantics=[('class', 'cube')],  position=(100, -200 , 100) )

    print("Getting Engine...")
    engine = rep.create.from_usd("/home/ubuntu/models/engine_color.usd",
                                 semantics=[('class', 'engine')], 
                                 count=1)

    with engine:
        rep.modify.pose(
            position=(0,0,500),
            scale = 1000
        )
    # print(f"After:\n{rep.utils.read_prim_transform(engine)}")
    spawn_volume = get_foreground_bound(config)

    # Generate N frames 
    with rep.trigger.on_frame(num_frames=config["num_frames"]):
        
        #  RANDOMIZE BACKGROUND PLANE
        with rep.create.group([plane]):
            rep.randomizer.color(
                colors=rep.distribution.uniform((0.2, 0.2, 0.2), (1, 1, 1))
            )
        
        # RANDOMIZE BACKGROUND OBJECTS
        #TODO: Add material/texture randomization
        with background_group:
            rep.randomizer.scatter_2d(plane,check_for_collisions=config["background"]["check_collisions"])
            rep.randomizer.color(
                colors=rep.distribution.uniform((0.2, 0.2, 0.2), (1, 1, 1))
            )
            rep.randomizer.rotation()

        # RANDOMIZE FOREGROUND OBJECTS
        with rep.create.group([engine]):
            # rep.randomizer.scatter_3d(spawn_volume)
            # rep.randomizer.scatter_2d(plane)
            # rep.randomizer.color(
            #     colors=rep.distribution.uniform((0.2, 0.2, 0.2), (1, 1, 1))
            # )
            rep.randomizer.rotation()
            # rep.modify.pose(
            #     position=rep.distribution.uniform((-200, -200, 200), (200, 200, 400)),
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

    print(f"Completed Scenario: {basename(output_dir)}")
    print("\n== DONE ==\n")