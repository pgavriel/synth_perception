import os 
import json

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
    for k, v in config.items():
        print(f"{k}: {v}")
    print("\n")
    
    return config
