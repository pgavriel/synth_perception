# Synth Perception   
Robotics perception utilizing synthetic data derrived from real object-centric data.  
![Banner Image](img/banner.png)  
  
This package supports the training of object detection and pose estimation models on synthetic data generated using [Unity Perception](https://github.com/Unity-Technologies/com.unity.perception). More recently, [Omniverse Replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html) has been adopted for data generation for improvements in both visual quality and ease of customization. The **replicator_docker** folder contains the necessary scripts for running Replicator inside a docker container and using it to generate custom synthetic data.  
Object models used in synthetic data generation are created via training NeRF models in [NerfStudio](https://docs.nerf.studio/) on object data generated using the [NIST MOAD Data Collection Rig](https://www.robot-manipulation.org/nist-moad).  
   
Progress thusfar includes scripts for converting the Unity Perception data format into [YOLO's Data Format](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format), training, and testing an object detection model trained on synthetic image data. Additionally, I have implemented a Pose Estimation model architecture based on the TQ-Net described in [Liu et al. 2020](https://ieeexplore.ieee.org/document/8868108), a script for converting Unity Perception data into a dataset for training this Pose Estimation model, and scripts for the training and subsequent testing of Pose Estimation models.  
   
### Using Omniverse Replicator for Data Generation   
#### replicator_docker/start_replicator.sh
I use an official Docker container from NVIDIA for running Replicator, which first requires [setting up an NGC API Key](https://org.ngc.nvidia.com/setup/api-key) and [logging into the container registry](https://docs.nvidia.com/ngc/gpu-cloud/ngc-catalog-user-guide/index.html#logging-in-to-ngc-registry). The container used is called [Synthetic Data Generation](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/ov-synthetic-data-generation?version=0.0.16-beta), and in their instructions, it's worth looking at the section "Accelerating Start up time" which will have you create a new container with a slightly different name so the **start_replicator.sh** script will need to be modified, but it will need to be modified no matter what to provide your local input/output directories for data generation. *INPUT_MODELS_DIR* should contain all of the USD models you wish to use as foreground objects, and *INPUT_MATERIALS_DIR* should contain all the image textures you would like to apply to foreground and background objects (they may be separated into subfolders to group different textures together). Once everything is configured properly you should be able to start the container:
``` 
cd synth_perception/replicator_docker
./start_replicator.sh
``` 
#### Generating Data inside the container
 Now that you are inside the container with all your data mounted properly, you need to edit a configuration file for data generation. Since the config folder is mounted, you can create new configs or edit existing ones and the changes will automatically appear in the container.   
 **Detailed information on config file parameters can be found [HERE](./replicator_docker/config/CONFIG_README.md)**  
 When you're ready to generate data, from the home directory of the conainter, you simply need to run the **run_batch.sh** script with any number of configs as command line arguments, which it will run in sequence, while logging all terminal output (NOTE: you do not need to include the .json extension in the arguments). For example:  
 ```
 // These are the same
 ./scripts/run_batch.sh dev_config
 ./scripts/run_batch.sh dev_config.json
 
// This will run three data generation batches in sequence
 ./scripts/run_batch.sh example_config example_config different_config
 ```
 The generated data will then be saved in the *LOCAL_OUTPUT_DIR* specified in **start_replicator.sh**.
   
### Important scripts in this repository are described below:  
#### Data Generation (Omniverse Replicator):  
##### replicator_to_yolo_dataset.py  
Combines an arbitrary number of synthetic data batches directly output from Replicator into a single training dataset formatted for directly training a YOLO detection model.   

#### Data Generation (Unity):  
##### unity_to_yolo_dataset.py  
Combines an arbitrary number of synthetic data batches directly output from Unity Perception, into a single training dataset formatted for directly training a YOLO detection model. It will first check each dataset provided for the object labels used, and combine them into a consistent master list so that datasets with different object annotations may easily be combined together. It is required that the Unity data contains annotations for 2D bounding boxes.  
  
##### unity_to_pose_dataset.py  
Combines an arbitrary number of synthetic data batches directly output from Unity Perception into a single training dataset formatted for the pose estimator model defined in **pose_estimator_model.py**. It is required that the Unity data contains annotations for both 2D and 3D bounding boxes.  
   
#### YOLO Object Detection:  
**yolo_train.py** - Loads a base YOLO model and a specified dataset generated by *unity_to_yolo_dataset.py*, and trains the model for a specified number of epochs.    
   
**yolo_test.py** - Loads a specified trained detection model and a list of image paths, runs each of the images through the detection model, draws the result, and saves the result image to a specified output directory.  
   
**yolo_live_demo.py** - Loads a specified trained detection model, opens a live camera feed to visualize object detections in real time. Additionally, draws the frame processing time in the corner.   
   
#### Pose Estimation:  
**pose_estimator_model.py** - Defines the network architecture of the Pose Estimation model using PyTorch. Also defines a custom data loader which is used to load datasets generated by *unity_to_pose_dataset.py*.  
   
**pose_estimator_train.py** - Trains a pose estimation model using a dataset generated by *unity_to_pose_dataset.py*. Automatically saves loss plots and logs training hyperparameters and other information to a csv log.  
    
**pose_estimator_test.py** - Similar to *yolo_test.py*, loads a set of images to pass through trained detection and pose estimation models, and saves the visualized result to an output directory.  
   
**pose_estimator_test_gt.py** - Similar to *pose_estimator_test.py*, but only expects to test on synthetic Unity datasets, and uses ground truth 2D bounding boxes as detections rather than relying on a detection model.  
   
**pose_estimator_live_demo.py** - Loads a specified detection model and pose estimation model, opens a live camera feed, and visualizes the output of each network in separate windows.  