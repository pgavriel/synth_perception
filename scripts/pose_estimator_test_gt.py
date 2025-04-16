from ultralytics import YOLO, settings
import cv2
import torch
import numpy as np
from os.path import join, isfile
import os 
import json
import utilities as util
from utilities import make_img_square, get_uvwh, get_image_paths, draw_3d_bounding_box, get_files, canonicalize_quaternion
from pose_estimator_model import PoseEstimationModel
from pose_estimator_train import geodesic_loss, rotation_angle_loss

"""
This script is used to validate the post estimation model on ground truth labels, using the ground truth 2D BB as the detection
and using the ground truth 3D BB as a ground truth comparison. 
"""
if __name__ == "__main__":
    # Load Object Size information
    object_sizes = util.load_json("object_sizes.json")

    save_images = True
    # output_dir = '/home/csrobot/Pictures/yolo_results_p'
    output_dir = util.create_incremental_dir("/home/csrobot/Pictures/","gt_gear_testing")
    os.makedirs(output_dir, exist_ok=True)

    visualize_images = False
    delay_ms = 200 # 0 to wait indefinitely for key input on each image

    # If we're testing on synthetic data, try to visualize the GT 3DBB
    visualize_synth_gt = True 

    # TODO: Determine programatically from camera GT
    FOCAL_LENGTH = 6172 #4000 #6172

    # Print Debug output
    verbose = False
    get_crops = False
    
    # Text options
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # White color
    thickness = 2
    # Load a detection model 
    # print("Loading Model...")
    # detect_model_folder = "train7"
    # detect_model_path = join("/home/csrobot/synth_perception/runs/detect",detect_model_folder,"weights/best.pt")
    # detect_model = YOLO(detect_model_path) # give path to .pt file

    # Display model information (optional)
    # model.info()

    # YOLO Inference Configuration [ https://docs.ultralytics.com/modes/predict/#inference-arguments ]
    min_conf = 0.5 # default: 0.25
    iou = 0.5 # default: 0.7 (lower numbers prevent prediction overlapping)
    visualize = False # default: False (saves a bunch of images)
    imgsz = 640 # default: 640 (width), 1920 on synth
    
    # Load the pose estimator model
    pose_model = PoseEstimationModel()
    model_folder = "mustard_041"
    model_folder = "gear_001"
    state_dict = torch.load(join("/home/csrobot/synth_perception/runs/pose_estimation",model_folder,"model_epoch_100.pth"), weights_only=True)
    pose_model.load_state_dict(state_dict)
    pose_model.eval()
    print('Pose Estimation model loaded successfully!')


    # Run inference test
    # Get a list of image paths to test the model on 
    # image_list = get_image_paths("/home/csrobot/Pictures/mustard_test")
    # image_list = sorted(image_list)
    image_list = get_files("/home/csrobot/Unity/SynthData/PoseTesting/gear/gear_n_d150")
    # image_list = get_files("/home/csrobot/Pictures/collected")
    image_list = image_list[:25]
    # image_list = [image_list[1]]
    print(f"Found {len(image_list)} images...")
    headers = ["LABEL","CONF","BOX (XYWH)","BOX (XYWHN)"]
    # For each test image...
    for c, img_path in enumerate(image_list, start=1):

        # Attempt to load image with opencv
        orig_img = cv2.imread(img_path)
        # orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        if orig_img is None:
            print(f"Error: Could not load {img_path}")
            continue
        img = orig_img.copy()
    

        # TODO: Load ground truth information for synthetic data image, and draw the ground truth 3dbb
        if visualize_synth_gt:
            # Making a few assumptions about file structure, should be direct output from Unity
            annotation_file = join(os.path.dirname(img_path),"step0.frame_data.json")
            if os.path.exists(annotation_file):
                with open(annotation_file, 'r') as f:
                    label_json = json.load(f)
                annotations = label_json["captures"][0]["annotations"]
                bb_3d = None
                bb_2d = None
                for ann in annotations:
                    if ann["id"] == "bounding box 3D":
                        if "values" in ann:
                            bb_3d = ann["values"]
                        else:
                            bb_3d = []
                        print(f"GROUND TRUTH 3DBB: {bb_3d}")
                    if ann["id"] == "bounding box":
                        if "values" in ann:
                            bb_2d = ann["values"]
                        else:
                            bb_2d = []
                        print(f"GROUND TRUTH 3DBB: {bb_2d}")

                losses = {}
                # For every pair of (2D/3D) ground truth bounding boxes found...
                for bb2d, bb3d in zip(bb_2d, bb_3d):
                    # Make sure 2D and 3D are for the same object instance
                    assert bb2d["instanceId"] == bb3d["instanceId"]
                    # Draw the ground truth 2D bounding box
                    xy1 = [int(bb2d["origin"][0]), int(bb2d["origin"][1])]
                    xy2 = [int(bb2d["origin"][0] + bb2d["dimension"][0]), int(bb2d["origin"][1] + bb2d["dimension"][1])]
                    img = cv2.rectangle(img,tuple(xy1),tuple(xy2),(255,0,0),1)
                    img = cv2.putText(img, str(bb2d["instanceId"]), tuple(xy1), font, font_scale, color, thickness)
                    # First, draw the ground truth bounding box
                    translation = bb3d["translation"]
                    size = bb3d["size"]
                    rotation = bb3d["rotation"]  # [x, y, z, w] quaternion
                    rotation = canonicalize_quaternion(rotation,False)
                    img = draw_3d_bounding_box(img, translation, size, rotation,fl=FOCAL_LENGTH, color=(255,255,255))
                    
                    # Extract the info from the ground truth to pass into the Pose Estimator
                    obj_label = bb2d["labelId"]
                    xyxy = np.asarray(xy1 + xy2)
                    pose_input_vector = get_uvwh(orig_img,obj_label,xyxy,FOCAL_LENGTH)
                    pose_input_vector = torch.tensor(pose_input_vector, dtype=torch.float32).unsqueeze(0) 
                    pose_input_crop = make_img_square(orig_img[xy1[1]:xy2[1],xy1[0]:xy2[0]]) # Crop [y1:y2,x1:x2]
                    pose_input_crop = torch.tensor(pose_input_crop, dtype=torch.float32).permute(2, 0, 1) / 255.0
                    pose_input_crop = pose_input_crop.unsqueeze(0) 
                    print(f"[Model Input Size [Crop]:{pose_input_crop.shape}   [Vector]:{pose_input_vector.shape}")
                    
                    # Pass input vectors through the pose estimator
                    with torch.no_grad():
                        output = pose_model(pose_input_crop, pose_input_vector).numpy()
                        # out_size = output[0][0:3] # Get size from json file instead
                        out_tran = output[0][3:6]
                        out_rot  = output[0][6:]
                    out_size = util.get_size_vector(bb2d["labelName"],object_sizes)
                    print(f"T: {translation}")
                    print(f"Model Output: \n[ SIZE ]{out_size}\n[ TRAN ]{out_tran}\n[ ROT  ]{out_rot}")
                    
                    # Visualize the pose estimator result
                    img = draw_3d_bounding_box(img, out_tran, out_size, out_rot, FOCAL_LENGTH)
                    
                    # Debug print out comparing result to ground truth
                    rotation = canonicalize_quaternion(rotation)
                    gt_rot_tensor = torch.tensor(rotation, dtype=torch.float32).unsqueeze(0) 
                    out_rot = canonicalize_quaternion(out_rot)
                    out_rot_tensor = torch.tensor(out_rot, dtype=torch.float32).unsqueeze(0) 
                    geo_loss = geodesic_loss(out_rot_tensor,gt_rot_tensor)
                    ang_loss = rotation_angle_loss(out_rot_tensor,gt_rot_tensor)
                    print(f"Geodesic Loss: {geo_loss:.2f}")
                    print(f"Angular Loss: {ang_loss:.2f}")
                    losses[str(bb2d["instanceId"])] = {"Inst":str(bb2d["instanceId"]),"Geo":geo_loss.item(), "Ang":ang_loss.item()}
                
                # Print Losses per detection
                for k in losses:
                    print(losses[k])

                # Show image?
                if visualize_images:
                    cv2.imshow("Out",img)
                    cv2.waitKey(delay_ms)
                # Save image?
                if save_images:
                    save_file = join(output_dir,f"result{c:03d}.jpg")
                    cv2.imwrite(save_file,img)
                    print(f"Result Saved: {save_file}")

       
        cv2.destroyAllWindows()
    print("Done")