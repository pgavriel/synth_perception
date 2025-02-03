# Synth Perception   
Robotics perception utilizing synthetic data derrived from real object-centric data.  
  
This package supports the training of object detection and pose estimation models on synthetic data generated using [Unity Perception](https://github.com/Unity-Technologies/com.unity.perception). 
Object models used in synthetic data generation are created via training NeRF models in [NerfStudio](https://docs.nerf.studio/) on object data generated using the [NIST MOAD Data Collection Rig](https://www.robot-manipulation.org/nist-moad).  
  
Progress thusfar includes scripts for converting the Unity Perception data format into [YOLO's Data Format](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format), training an object detection model, and testing that model on real world images.  
  