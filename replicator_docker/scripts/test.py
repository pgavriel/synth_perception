import omni.replicator.core as rep

import os.path
from os.path import join, dirname
import sys

sys.path.append(join(dirname(__file__), '..'))
sys.path.append(dirname(__file__))
import utilities as util

config_folder = "/home/ubuntu/config"
config = util.load_json_config(join(config_folder,"config_a.json"))

output_root = config["output_root"]
output_dir = util.create_incremental_dir(output_root,config["output_name"])

# New Scenario
with rep.new_layer():

    # Create Camera
    camera = rep.create.camera(position=(0, 0, 0),look_at=(0,0,10))

    # Create Lighting (Create function for different types)
    sphere_light = rep.create.light(
        light_type="Sphere",
        temperature=rep.distribution.normal(6500, 500),
        intensity=rep.distribution.normal(35000, 5000),
        position=rep.distribution.uniform((-300, -300, -300), (300, 300, 300)),
        scale=rep.distribution.uniform(50, 100),
        count=2
    )

    # Output render size
    render_product = rep.create.render_product(camera, tuple(config["output_size"]))

    # Create Foreground objects
    torus = rep.create.torus(semantics=[('class', 'torus')] , position=(0, -200 , 100))

    sphere = rep.create.sphere(semantics=[('class', 'sphere')], position=(0, 100, 100))

    cube = rep.create.cube(semantics=[('class', 'cube')],  position=(100, -200 , 100) )

    # Generate N frames 
    with rep.trigger.on_frame(num_frames=10):
        with rep.create.group([torus, sphere, cube]):
            rep.modify.pose(
                position=rep.distribution.uniform((-200, -200, 200), (200, 200, 400)),
                scale=rep.distribution.uniform(0.1, 2))

    # Initialize and attach writer
    writer = rep.WriterRegistry.get("BasicWriter")

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