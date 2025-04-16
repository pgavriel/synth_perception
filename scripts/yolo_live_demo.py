import cv2
import time
from ultralytics import YOLO
from os.path import join

# Load the YOLO model
# Load a model
model_folder = "gear2"
model_path = join("/home/csrobot/synth_perception/runs/detect",model_folder,"weights/best.pt")
model = YOLO(model_path)  
model.info()

# YOLO Inference Configuration [ https://docs.ultralytics.com/modes/predict/#inference-arguments ]
min_conf = 0.5 # default: 0.25
iou = 0.5 # default: 0.7 (lower numbers prevent prediction overlapping)
max_det = 1
visualize = False # default: False (saves a bunch of images)
imgsz = (480,640) # default: 640 (width), 1920 on synth

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
        results = model(frame, imgsz=imgsz, visualize=visualize, conf=min_conf, iou=iou, max_det=max_det)

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

        # Rescale the image for display
        scale_factor = 1.5
        height, width = annotated_frame.shape[:2]
        new_width = int(width * scale_factor)  
        new_height = int(height * scale_factor) 
        annotated_frame = cv2.resize(annotated_frame, (new_width, new_height))

        # Display the resulting frame
        cv2.imshow('YOLO Live Detection', annotated_frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
