import cv2
import time
import torch
from ultralytics import YOLO
from os.path import join
from pose_estimator_model import PoseEstimationModel

# Load the YOLO detection model
model_folder = "train7"
model_path = join("/home/csrobot/synth_perception/runs/detect",model_folder,"weights/best.pt")
detect_model = YOLO(model_path)  
detect_model.info()

# Load the pose estimator model
pose_model = PoseEstimationModel()
model_folder = "mustard_003"
state_dict = torch.load(join("/home/csrobot/synth_perception/runs/pose_estimation",model_folder,"model_epoch_25.pth"), weights_only=True)
pose_model.load_state_dict(state_dict)
print('Weights loaded successfully!')

exit()
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
        results = detect_model(frame)

        # Draw the detections on the frame
        annotated_frame = results[0].plot()  # Automatically draws boxes and labels

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
