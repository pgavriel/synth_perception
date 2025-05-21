# Step 1. Set up NGC API Key: https://org.ngc.nvidia.com/setup/api-key 
# Step 2. Log in to the container registry: https://docs.nvidia.com/ngc/gpu-cloud/ngc-catalog-user-guide/index.html#logging-in-to-ngc-registry 
# Step 3. Configure volume paths to mount folders for scripts, output, configs, models, and textures

docker run --gpus all --entrypoint /bin/bash -it \
    -v "/home/csrobot/synth_perception/replicator_docker/scripts:/home/ubuntu/scripts" \
    -v "/home/csrobot/Omniverse/SynthData:/home/ubuntu/output" \
    -v "/home/csrobot/synth_perception/replicator_docker/config:/home/ubuntu/config" \
    -v "/home/csrobot/Omniverse/Models:/home/ubuntu/models" \
    -v "/home/csrobot/Omniverse/Materials/vMaterials_2:/home/ubuntu/materials" \
    ov-synthetic-data-generation-startup:v1

# Original image name    
# nvcr.io/nvidia/ov-synthetic-data-generation:0.0.16-beta