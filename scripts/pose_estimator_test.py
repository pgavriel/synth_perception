from ultralytics import YOLO, settings
import cv2
import torch
from os.path import join, isfile
import os 
import json
from utilities import make_img_square, get_uvwh, get_image_paths, draw_3d_bounding_box, get_files
from pose_estimator_model import PoseEstimationModel

if __name__ == "__main__":
    save_images = True
    output_dir = '/home/csrobot/Pictures/yolo_results_j'
    os.makedirs(output_dir, exist_ok=True)

    visualize_images = True
    delay_ms = 200 # 0 to wait indefinitely for key input on each image

    # If we're testing on synthetic data, try to visualize the GT 3DBB
    visualize_synth_gt = True 
    # draw_corners = True
    # draw_edges = False

    # Print Debug output
    verbose = False
    get_crops = False

    # Load a detection model 
    print("Loading Model...")
    detect_model_folder = "train7"
    detect_model_path = join("/home/csrobot/synth_perception/runs/detect",detect_model_folder,"weights/best.pt")
    detect_model = YOLO(detect_model_path) # give path to .pt file

    # Display model information (optional)
    # model.info()

    # YOLO Inference Configuration [ https://docs.ultralytics.com/modes/predict/#inference-arguments ]
    min_conf = 0.5 # default: 0.25
    iou = 0.5 # default: 0.7 (lower numbers prevent prediction overlapping)
    visualize = False # default: False (saves a bunch of images)
    imgsz = 1920 # default: 640 (width)
    
    # Load the pose estimator model
    pose_model = PoseEstimationModel()
    model_folder = "mustard_013"
    state_dict = torch.load(join("/home/csrobot/synth_perception/runs/pose_estimation",model_folder,"model_epoch_100.pth"), weights_only=True)
    pose_model.load_state_dict(state_dict)
    pose_model.eval()
    print('Pose Estimation model loaded successfully!')


    # Run inference with the trained YOLO model
    # Get a list of image paths to test the model on 
    # image_list = get_image_paths("/home/csrobot/Pictures/mustard_test")
    # image_list = sorted(image_list)
    image_list = get_files("/home/csrobot/Unity/SynthData/PoseTesting/mustard_nerve")
    image_list = image_list[:50]
    # image_list = [image_list[1]]
    print(f"Found {len(image_list)} images...")
    headers = ["LABEL","CONF","BOX (XYWH)","BOX (XYWHN)"]
    # For each test image...
    for c, img_path in enumerate(image_list, start=1):

        # Attempt to load image with opencv
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load {img_path}")
            continue
        
        # TODO: Load ground truth information for synthetic data image, and draw the ground truth 3dbb
        if visualize_synth_gt:
            # Making a few assumptions about file structure, should be direct output from Unity
            annotation_file = join(os.path.dirname(img_path),"step0.frame_data.json")
            if os.path.exists(annotation_file):
                with open(annotation_file, 'r') as f:
                    label_json = json.load(f)
                annotations = label_json["captures"][0]["annotations"]
                bb_3d = None
                for ann in annotations:
                    if ann["id"] == "bounding box 3D":
                        if "values" in ann:
                            bb_3d = ann["values"]
                            for bbox in bb_3d:
                                translation = bbox["translation"]
                                size = bbox["size"]
                                rotation = bbox["rotation"]  # [x, y, z, w] quaternion
                                img = draw_3d_bounding_box(img, translation, size, rotation,fl=6172, color=(255,255,255))
                        else:
                            bb_3d = None
                        print(f"GROUND TRUTH 3DBB: {bb_3d}")
                # Save image before passing through model?
                save_file = join(output_dir,f"result{c:03d}.jpg")
                cv2.imwrite(save_file,img)
                print(f"Result Saved: {save_file}")

        # If successful, pass the image through the detection model (returns a list)
        results = detect_model(img, imgsz=imgsz, visualize=visualize, conf=min_conf, iou=iou)

        # For each result (should be length 1 for evaluating a single image)
        for result in results:  
            # If there are detection boxes associated with result... 
            # print(f"Classes: {result.names}")
            res = result.boxes.cpu().numpy()
            if len(res) > 0:
                # Print out some info about the detection bounding box
                print(f"Original Image Size (WxH): {res.orig_shape[1]} x {res.orig_shape[0]}")
                print(f"[{headers[0].center(10)}][{headers[1].center(10)}][{headers[2].center(49)}][{headers[3].center(49)}]")
                crops = []
                for d, xywh, xywhn, xyxy in zip(res.data, res.xywh, res.xywhn, res.xyxy):
                    conf = f"{d[4]:.3f}"
                    print(f"[{result.names[d[5]].center(10)}][{conf.center(10)}][{xywh}][{xywhn}]")
                    print(f"D: {d}")
                    FOCAL_LENGTH = 6172
                    pose_input_vector = get_uvwh(img,d[5],xyxy,FOCAL_LENGTH)
                    x1, y1, x2, y2 = map(int, xyxy)
                    pose_input_crop = make_img_square(result.orig_img[y1:y2,x1:x2])
                    # print(f"[Img: {type(pose_input_crop)} {pose_input_crop.shape}][Vector: {type(pose_input_vector)} {pose_input_vector.shape}]")
                    pose_input_vector = torch.tensor(pose_input_vector, dtype=torch.float32).unsqueeze(0) 
                    pose_input_crop = torch.tensor(pose_input_crop, dtype=torch.float32).permute(2, 0, 1) / 255.0
                    pose_input_crop = pose_input_crop.unsqueeze(0) 
                    print(f"[Model Input Size [Crop]:{pose_input_crop.shape}   [Vector]:{pose_input_vector.shape}")
                    with torch.no_grad():
                        output = pose_model(pose_input_crop, pose_input_vector).numpy()
                    out_size = output[0][0:3]
                    out_tran = output[0][3:6]
                    out_rot  = output[0][6:]
                    # pose_outputs = model(pose_input_crop, pose_input_vector)
                    # print(f"Model Output: {output}")
                    print(f"Model Output: \n[ SIZE ]{out_size}\n[ TRAN ]{out_tran}\n[ ROT  ]{out_rot}")
                    
                    # pose_image = draw_3d_bounding_box(img, out_size, out_tran, out_rot,FOCAL_LENGTH)
                    pose_image = draw_3d_bounding_box(img, out_tran, out_size, out_rot,FOCAL_LENGTH)
                    crops.append(make_img_square(result.orig_img[y1:y2,x1:x2]))
                    window_name = str(len(crops))
                    cv2.imshow(window_name, crops[-1])

                    cv2.imshow("Pose",pose_image)
                # print(result.boxes.cpu().numpy())
                # cv2.waitKey(0)
                if save_images: # Save annotated image
                    save_file = join(output_dir,f"result{c:03d}.jpg")
                    result.save(filename=save_file)  
                    print(f"Result Saved: {save_file}")
            else:
                 print("No detections found.")

            # if save_images: # Save annotated image
            #         save_file = join(output_dir,f"result{c:03d}.jpg")
            #         result.save(filename=save_file)  
            #         print(f"Result Saved: {save_file}")
                
            # Visualize the detection
            if visualize_images:
                res_img = result.plot()  # This plots the detections on the image
                cv2.imshow('Detection Results', res_img)
                cv2.waitKey(delay_ms)

        cv2.destroyAllWindows()
    print("Done")