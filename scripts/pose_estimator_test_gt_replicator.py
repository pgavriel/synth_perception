from ultralytics import YOLO, settings
import cv2
import torch
import numpy as np
from os.path import join, isfile
import os 
import json
import glob
import utilities as util
from utilities import make_img_square, get_uvwh, get_image_paths, draw_3d_bounding_box, get_files, canonicalize_quaternion, replicator_extract_3dbb_info, replicator_draw_3d_bounding_box
from pose_estimator_model import PoseEstimationModel
from pose_estimator_train import geodesic_loss, rotation_angle_loss

"""
This script is used to validate the post estimation model on ground truth labels, using the ground truth 2D BB as the detection
and using the ground truth 3D BB as a ground truth comparison. 
"""
if __name__ == "__main__":
    # Load Object Size information
    object_sizes = util.load_json("object_sizes.json")

    save_images = False
    # output_dir = '/home/csrobot/Pictures/yolo_results_p'
    if save_images:
        output_dir = util.create_incremental_dir("/home/csrobot/Pictures/","rep_pose_testing")
        os.makedirs(output_dir, exist_ok=True)

    visualize_images = True
    delay_ms = 0 # 0 to wait indefinitely for key input on each image

    # If we're testing on synthetic data, try to visualize the GT 3DBB
    visualize_synth_gt = True 

    # TODO: Determine programatically from camera GT
    FOCAL_LENGTH = 6172 #4000 #6172

    # Print Debug output
    verbose = False
    get_crops = False
    
    # Text options
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
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
    # min_conf = 0.5 # default: 0.25
    # iou = 0.5 # default: 0.7 (lower numbers prevent prediction overlapping)
    # visualize = False # default: False (saves a bunch of images)
    # imgsz = 640 # default: 640 (width), 1920 on synth
    
    # Load the pose estimator model
    pose_model = PoseEstimationModel()
    # model_folder = "mustard_041"
    model_folder = "replicator_engine_009"
    state_dict = torch.load(join("/home/csrobot/synth_perception/runs/pose_estimation",model_folder,"model_epoch_100.pth"), weights_only=True)
    pose_model.load_state_dict(state_dict)
    pose_model.eval()
    print('Pose Estimation model loaded successfully!')

    image_pattern="rgb_{}.png"
    bb_2d_style = "loose" # "tight or "loose"
    bb_2d_npy_pattern="bounding_box_2d_{}_{}.npy"
    bb_2d_json_pattern="bounding_box_2d_{}_labels_{}.json"
    bb_3d_npy_pattern="bounding_box_3d_{}.npy"
    bb_3d_json_pattern="bounding_box_3d_labels_{}.json"
    # Run inference test
    # Get a list of image paths to test the model on 
    # image_list = get_image_paths("/home/csrobot/Pictures/mustard_test")
    # image_list = sorted(image_list)
    # image_list = get_files("/home/csrobot/Omniverse/SynthData/engine/test_001")
    folder_path = "/home/csrobot/Omniverse/SynthData/engine/loose_001"
    image_files = sorted(glob.glob(os.path.join(folder_path, image_pattern.format("*"))))

    # image_list = get_files("/home/csrobot/Pictures/collected")
    # image_files = image_files[:3]
    # image_list = [image_list[1]]
    print(f"Found {len(image_files)} images...")
    headers = ["LABEL","CONF","BOX (XYWH)","BOX (XYWHN)"]
    # exit()
    # For each test image...
    for c, img_path in enumerate(image_files, start=1):
        # Extract frame number
        frame_num = os.path.splitext(os.path.basename(img_path))[0].split("_")[-1]
        if int(frame_num) == 0: continue

        # Get annotation file paths
        bb_2d_npy_path = os.path.join(folder_path, bb_2d_npy_pattern.format(bb_2d_style,frame_num))
        bb_2d_json_path = os.path.join(folder_path, bb_2d_json_pattern.format(bb_2d_style,frame_num))
        bb_3d_npy_path = os.path.join(folder_path, bb_3d_npy_pattern.format(frame_num))
        bb_3d_json_path = os.path.join(folder_path, bb_3d_json_pattern.format(frame_num))
        # Verify each component exists 
        if not os.path.exists(bb_2d_npy_path):
            print(f"ERROR: Missing .npy file for frame {frame_num}: {bb_2d_npy_path}")
            continue
        if not os.path.exists(bb_2d_json_path):
            print(f"ERROR: Missing .json file for frame {frame_num}: {bb_2d_json_path}")
            continue
        if not os.path.exists(bb_3d_npy_path):
            print(f"ERROR: Missing .npy file for frame {frame_num}: {bb_3d_npy_path}")
            continue
        if not os.path.exists(bb_3d_json_path):
            print(f"ERROR: Missing .json file for frame {frame_num}: {bb_3d_json_path}")
            continue
        if not os.path.exists(img_path):
            print(f"ERROR: Image {img_path} not found, skipping.")
            continue

        # Attempt to load image with opencv
        orig_img = cv2.imread(img_path)
        # orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        if orig_img is None:
            print(f"Error: Could not load {img_path}")
            continue
        img = orig_img.copy()

        # GET CAMERA INTRINSIC INFORMATION
        #TODO: These values are currently hardcoded, but could be extracted from camera params files
        FOCAL_LENGTH = 2199 # Represents focal length X and Y
        w = 1920
        h = 1080
        cam_cx = w / 2
        cam_cy = h /2
        cam_fx = FOCAL_LENGTH * cam_cx # camera matrix[0][0]
        cam_fy = FOCAL_LENGTH * cam_cy # camera matrix[1][1]
    
        #Labels for 2D and 3D should be the same for any given dataset
        with open(bb_2d_json_path, "r") as f:
            class_labels_2d = json.load(f)
        with open(bb_3d_json_path, "r") as f:
            class_labels_3d = json.load(f)
        # print(f"Labels: {labels}")

        bboxes_2d = np.load(bb_2d_npy_path)
        bboxes_3d = np.load(bb_3d_npy_path)
        
        if len(bboxes_2d) != len(bboxes_3d):
            print(f"SKIP! - Len 2D: {len(bboxes_2d)} - Len 3D: {len(bboxes_3d)} - {bb_2d_json_path}")
            continue
        
        # Load Annotations (Replicator)
        labels = replicator_extract_3dbb_info(annotation_file=bb_3d_npy_path,verbose=True)
    

        losses = {}
        instance_id = 1
        # For every pair of (2D/3D) ground truth bounding boxes found...
        for bb2d, bb3d, labels3d in zip(bboxes_2d, bboxes_3d, labels):
            print(f"\n===== [ IMAGE {c}][ DETECTION {instance_id}] ==========================================")
            # VISUALIZE GROUND TRUTHS  ======================
            # Draw the ground truth 2D bounding box
            id, x1, y1, x2, y2, occlusion = bb2d
            pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(img, pt1, pt2, color, 1)
            img = cv2.putText(img, str(id), (x1-5,y1-5), font, font_scale, color, thickness)
            # Then, draw the 3D ground truth bounding box
            translation = labels3d[1]
            rotation = labels3d[2]
            size = labels3d[3]
            rotation = canonicalize_quaternion(rotation,False)
            img = replicator_draw_3d_bounding_box(img, translation, size, rotation,FOCAL_LENGTH,color=(255,255,255),verbose=False)
            
            # RUN MODEL INFERENCE =========================
            # # Extract the info from the ground truth to pass into the Pose Estimator
            xyxy = np.asarray([x1, y1, x2, y2])
            pose_input_vector = get_uvwh(orig_img,id,xyxy,FOCAL_LENGTH,verbose=False)
            print(f"Model Input Vector LUVWH: {[float(p) for p in pose_input_vector]}")
            pose_input_vector = torch.tensor(pose_input_vector, dtype=torch.float32).unsqueeze(0) 
            pose_input_crop = make_img_square(orig_img[y1:y2,x1:x2]) # Crop [y1:y2,x1:x2]
            pose_input_crop = torch.tensor(pose_input_crop, dtype=torch.float32).permute(2, 0, 1) / 255.0
            pose_input_crop = pose_input_crop.unsqueeze(0) 
            print(f"Model Input Size [Crop]:{pose_input_crop.shape}   [Vector]:{pose_input_vector.shape}")
            # print(f"\nGround Truth 3D Vectors:\n[ SIZE ]{size}\n[ TRAN ]{translation}\n[ ROT  ]{[float(r) for r in rotation]}")

            # # Pass input vectors through the pose estimator
            with torch.no_grad():
                output = pose_model(pose_input_crop, pose_input_vector).numpy()
                # out_size = output[0][0:3] # Get size from json file instead
                out_tran = output[0][3:6]
                out_rot  = output[0][6:]
            out_size = util.get_size_vector("engine",object_sizes)
            # print(f"T: {translation}")
            # print(f"Model Output: \n[ SIZE ]{out_size}\n[ TRAN ]{out_tran}\n[ ROT  ]{out_rot}")
            
            # Debug output, show comparison between GT and Model estimate
            print("COMPARISON (Ground Truth vs Model Estimate):")
            print(f"\n[ SIZE  ]\n[  GT   ] {size}\n[ MODEL ] {out_size}")
            print(f"\n[ TRANSLATION ]\n[  GT   ] {translation}\n[ MODEL ] {out_tran}")
            print(f"[ DIFF  ] {np.array([float(f'{ev - gt:.2f}') for ev, gt in zip(out_tran,translation)])}")
            distance = np.linalg.norm(translation - out_tran)
            print("Euclidean Distance:", distance)
            
            print(f"\n[ ROTATION ]\n[  GT   ] {np.array([float(f'{r:.5f}') for r in rotation])}\n[ MODEL ] {out_rot}")
            rot_diff = np.array([float(f'{ev - gt:.5f}') for ev, gt in zip(out_rot,rotation)])
            print(f"[ DIFF  ] {rot_diff} [ ABS SUM ] = {np.sum(abs(rot_diff))}")

            # # Visualize the pose estimator result
            img = replicator_draw_3d_bounding_box(img, out_tran, out_size, out_rot,FOCAL_LENGTH,color=(0,255,255),verbose=False)
            
            # # Debug print out comparing result to ground truth
            rotation = canonicalize_quaternion(rotation)
            gt_rot_tensor = torch.tensor(rotation, dtype=torch.float32).unsqueeze(0) 
            out_rot = canonicalize_quaternion(out_rot)
            out_rot_tensor = torch.tensor(out_rot, dtype=torch.float32).unsqueeze(0) 
            geo_loss = geodesic_loss(out_rot_tensor,gt_rot_tensor)
            ang_loss = rotation_angle_loss(out_rot_tensor,gt_rot_tensor)
            
            print(f"\nGeodesic Loss: {geo_loss:.2f}")
            print(f"Angular Loss: {ang_loss:.2f}")
            losses[str(instance_id)] = {"Inst":str(instance_id),"Geo":geo_loss.item(), "Ang":ang_loss.item()}
            
            instance_id += 1 
        # Print Losses per detection
        for k in losses:
            print(losses[k])


        # Show image?
        if visualize_images:
            cv2.imshow("Out",img)
            k = cv2.waitKey(delay_ms)
            if k == ord('q'):
                break
        # Save image?
        if save_images:
            save_file = join(output_dir,f"result{c:03d}.jpg")
            cv2.imwrite(save_file,img)
            print(f"Result Saved: {save_file}")

       
        cv2.destroyAllWindows()
    print("Done")