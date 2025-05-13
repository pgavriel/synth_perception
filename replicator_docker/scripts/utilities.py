import os 
import json
import math

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def create_incremental_dir(root, prefix="test", digits=3):
    os.makedirs(root, exist_ok=True)  # Ensure root exists
    i = 1
    while True:
        new_dir = os.path.join(root, f"{prefix}_{i:0{digits}}")
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            return new_dir
        i += 1

def load_json_config(file):
    with open(file, "r") as f:
        config = json.load(f)

    print("\nLoaded config:")
    # for k, v in config.items():
    #     print(f"{k}: {v}")
    print(json.dumps(config,sort_keys=False,indent=2))
    print("\n")
    
    return config

def compute_plane_size(fov_deg, aspect_ratio, depth):
    fov_rad = math.radians(fov_deg)
    height = 2 * depth * math.tan(fov_rad / 2)
    width = height * aspect_ratio
    return width, height

def print_object_info(object,object_name=None):
    print(f"Object Info: {object_name}")
    print(f"Type: {type(object)}")
    print(f"Dir: {dir(object)}")