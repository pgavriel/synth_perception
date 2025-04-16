import cv2
import time
import torch
from ultralytics import YOLO
from os.path import join
from pose_estimator_model import PoseEstimationModel
import utilities as util
from utilities import make_img_square, get_uvwh, get_image_paths, draw_3d_bounding_box, get_files

# Load the YOLO detection model
model_folder = "train7" # Mustard Testing
# model_folder = "gear2" # Gear Testing
model_path = join("/home/csrobot/synth_perception/runs/detect",model_folder,"weights/best.pt")
detect_model = YOLO(model_path)  
detect_model.info()

# YOLO Inference Configuration [ https://docs.ultralytics.com/modes/predict/#inference-arguments ]
min_conf = 0.5 # default: 0.25
iou = 0.5 # default: 0.7 (lower numbers prevent prediction overlapping)
max_det = 1
visualize = False # default: False (saves a bunch of images)
imgsz = (480,640) # default: 640 (width), 1920 on synth
visualize_detection = True # Whether to visualize detection

# Load the pose estimator model
pose_model = PoseEstimationModel()
model_folder = "mustard_041" # Mustard Testing
# model_folder = "gear_001" # Gear Testing
model_name = "model_epoch_100.pth"
state_dict = torch.load(join("/home/csrobot/synth_perception/runs/pose_estimation",model_folder,model_name), weights_only=True)
pose_model.load_state_dict(state_dict)
print('Weights loaded successfully!')

headers = ["LABEL","CONF","BOX (XYWH)","BOX (XYWHN)"]

# Load Object Size JSON
object_sizes = util.load_json("object_sizes.json")

cv2.namedWindow('YOLO Live Detection', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Pose Estimation', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Detection Crop', cv2.WINDOW_AUTOSIZE)
# exit()
# Initialize the video capture (camera device 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Start time for processing
        start_time = time.time()

        # Perform object detection
        # results = detect_model(frame)


        # If successful, pass the image through the detection model (returns a list)
        results = detect_model(frame, imgsz=imgsz, visualize=visualize, conf=min_conf, iou=iou, max_det=max_det)

        # Draw the detections on the frame
        annotated_frame = results[0].plot()  # Automatically draws boxes and labels
        
            # cv2.imshow('Detection Results', res_img)
            # cv2.waitKey(delay_ms)

        # For each result (should be length 1 for evaluating a single image)
        for result in results:  
            # Visualize the detection
            if visualize_detection:
                frame = result.plot()  # This plots the detections on the image
            # If there are detection boxes associated with result... 
            # print(f"Classes: {result.names}")
            res = result.boxes.cpu().numpy()
            if len(res) > 0:
                # Print out some info about the detection bounding box
                print(f"Original Image Size (WxH): {res.orig_shape[1]} x {res.orig_shape[0]}")
                print(f"[{headers[0].center(10)}][{headers[1].center(10)}][{headers[2].center(49)}][{headers[3].center(49)}]")
                crops = []
                names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
                for d, xywh, xywhn, xyxy, name in zip(res.data, res.xywh, res.xywhn, res.xyxy, names):
                    conf = f"{d[4]:.3f}"
                    print(f"[{result.names[d[5]].center(10)}][{conf.center(10)}][{xywh}][{xywhn}]")
                    print(f"D: {d}")
                    FOCAL_LENGTH = 4000 #4000 #6172 #429?
                    pose_input_vector = get_uvwh(frame,d[5],xyxy,FOCAL_LENGTH)
                    x1, y1, x2, y2 = map(int, xyxy)
                    pose_input_crop = make_img_square(result.orig_img[y1:y2,x1:x2])
                    # print(f"[Img: {type(pose_input_crop)} {pose_input_crop.shape}][Vector: {type(pose_input_vector)} {pose_input_vector.shape}]")
                    pose_input_vector = torch.tensor(pose_input_vector, dtype=torch.float32).unsqueeze(0) 
                    pose_input_crop = torch.tensor(pose_input_crop, dtype=torch.float32).permute(2, 0, 1) / 255.0
                    pose_input_crop = pose_input_crop.unsqueeze(0) 
                    print(f"[Model Input Size [Crop]:{pose_input_crop.shape}   [Vector]:{pose_input_vector.shape}")
                    
                    
                    with torch.no_grad():
                        output = pose_model(pose_input_crop, pose_input_vector).numpy()
                    # out_size = output[0][0:3]
                    print("NAME ",name)
                    out_size = util.get_size_vector(name,object_sizes)
                    out_tran = output[0][3:6]
                    out_rot  = output[0][6:]

                    # pose_outputs = model(pose_input_crop, pose_input_vector)
                    # print(f"Model Output: {output}")
                    print(f"Model Output: \n[ SIZE ]{out_size}\n[ TRAN ]{out_tran}\n[ ROT  ]{out_rot}")
                    
                    # pose_image = draw_3d_bounding_box(img, out_size, out_tran, out_rot,FOCAL_LENGTH)

                    crops.append(make_img_square(result.orig_img[y1:y2,x1:x2]))
                    pose_image = draw_3d_bounding_box(result.orig_img, out_tran, out_size, out_rot,FOCAL_LENGTH)
                    window_name = str(len(crops))
                    cv2.imshow("Detection Crop", crops[-1])

                    cv2.imshow("Pose Estimation",pose_image)

            else:
                 print("No detections found.")

     
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Add processing time text to the frame
        text = f"{processing_time:.2f} ms"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)  # Green
        thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = 10
        text_y = frame.shape[0] - 10
        cv2.putText(annotated_frame, text, (text_x+2, text_y+2), font, font_scale, (0,0,0), thickness)
        cv2.putText(annotated_frame, text, (text_x, text_y), font, font_scale, color, thickness)

        # Display the resulting frame
        cv2.imshow('YOLO Live Detection', annotated_frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
