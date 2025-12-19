# Step 1. Set up NGC API Key: https://org.ngc.nvidia.com/setup/api-key 
# Step 2. Log in to the container registry: https://docs.nvidia.com/ngc/gpu-cloud/ngc-catalog-user-guide/index.html#logging-in-to-ngc-registry 
# Step 3. Configure volume paths to mount folders for scripts, output, configs, models, and textures

# Official Container: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/ov-synthetic-data-generation?version=0.0.16-beta 

# Modify local input/output directories
INPUT_MODELS_DIR="/home/csrobot/Omniverse/Models"
INPUT_MATERIALS_DIR="/home/csrobot/Omniverse/Materials/vMaterials_2"
LOCAL_OUTPUT_DIR="/home/csrobot/Omniverse/SynthData"

LOCAL_SCRIPT_DIR=$(realpath ./scripts)
LOCAL_CONFIG_DIR=$(realpath ./config)
CONTAINER_NAME="nvcr.io/nvidia/ov-synthetic-data-generation:0.0.16-beta"

echo "INPUT_MODELS_DIR:    $INPUT_MODELS_DIR"
echo "INPUT_MATERIALS_DIR: $INPUT_MATERIALS_DIR"
echo "LOCAL_OUTPUT_DIR:    $LOCAL_OUTPUT_DIR"
echo "LOCAL_SCRIPT_DIR:    $LOCAL_SCRIPT_DIR"
echo "LOCAL_CONFIG_DIR:    $LOCAL_CONFIG_DIR"
sleep 1

echo "Starting container: $CONTAINER_NAME..."
docker run --gpus all --entrypoint /bin/bash -it \
    -v "$LOCAL_SCRIPT_DIR:/home/ubuntu/scripts" \
    -v "$LOCAL_OUTPUT_DIR:/home/ubuntu/output" \
    -v "$LOCAL_CONFIG_DIR:/home/ubuntu/config" \
    -v "$INPUT_MODELS_DIR:/home/ubuntu/models" \
    -v "$INPUT_MATERIALS_DIR:/home/ubuntu/materials" \
    $CONTAINER_NAME

# Original container name    
# nvcr.io/nvidia/ov-synthetic-data-generation:0.0.16-beta

# Custom container name (after following "Accelerating Start up time" instructions)
# ov-synthetic-data-generation-startup:v1