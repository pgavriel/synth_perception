from ultralytics import YOLO, settings
import cv2
from os.path import join, isfile
import os 

def make_img_square(img, size=96, verbose=False):
    #TODO: Implement padding with noise rather than flat color
    #TODO: Implement options for resize interpolation

    h,w,_ = img.shape
    if verbose: print(f"Passed Image Shape: {img.shape}")
   
    # Determine padding for height and width
    pad_top = pad_bottom = (max(h, w) - h) // 2
    pad_left = pad_right = (max(h, w) - w) // 2
    
    # If the difference is odd, add an extra pixel of padding to the bottom/right
    pad_bottom += (max(h, w) - h) % 2
    pad_right += (max(h, w) - w) % 2

    # Pad the image with a solid color
    pad_color = (0,0,0)
    square_img = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=pad_color
    )   

    # Resize 
    resized_image = cv2.resize(square_img, (size, size))
    
    return resized_image


def get_image_paths(directory):
    """Returns a list of image paths found inside directory."""

    # Define a set of common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    # Get full paths for all files with image extensions in the target directory
    image_paths = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if os.path.splitext(file)[1].lower() in image_extensions
    ]
    return image_paths

if __name__ == "__main__":
    save_images = False
    output_dir = '/home/csrobot/images/yolo_results'
    os.makedirs(output_dir, exist_ok=True)

    visualize_images = False
    delay_ms = 500 # 0 to wait indefinitely for key input on each image

    # Print Debug output
    verbose = True
    get_crops = True

    # Load a model 
    model_folder = "train4"
    model_path = join("/home/csrobot/synth_perception/runs/detect",model_folder,"weights/best.pt")
    model = YOLO(model_path) # give path to .pt file

    # Display model information (optional)
    # model.info()

    # Inference Configuration [ https://docs.ultralytics.com/modes/predict/#inference-arguments ]
    min_conf = 0.5 # default: 0.25
    iou = 0.5 # default: 0.7 (lower numbers prevent prediction overlapping)
    visualize = False # default: False (saves a bunch of images)
    imgsz = 1920 # default: 640 (width)
    
    # Run inference with the trained YOLO model
    # Get a list of image paths to test the model on 
    image_list = get_image_paths("/home/csrobot/ns-data/engine-18/DSLR/")
    image_list = sorted(image_list)
    image_list = ["/home/csrobot/synth_perception/engine_test.png"]
    headers = ["LABEL","CONF","BOX (XYWH)","BOX (XYWHN)"]
    for c, img_path in enumerate(image_list, start=1):
        img = img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load {img_path}")
            continue
        results = model(img, imgsz=imgsz, visualize=visualize, conf=min_conf, iou=iou)
        
        for result in results:    
            if verbose: 
                # print(f"Classes: {result.names}")
                res = result.boxes.cpu().numpy()
                print(f"Original Image Size (WxH): {res.orig_shape[1]} x {res.orig_shape[0]}")
                print(f"[{headers[0].center(10)}][{headers[1].center(10)}][{headers[2].center(49)}][{headers[3].center(49)}]")
                crops = []
                for d, xywh, xywhn, xyxy in zip(res.data, res.xywh, res.xywhn, res.xyxy):
                    x1, y1, x2, y2 = map(int, xyxy)
                    crops.append(make_img_square(result.orig_img[y1:y2,x1:x2]))
                    conf = f"{d[4]:.3f}"
                    print(f"[{result.names[d[5]].center(10)}][{conf.center(10)}][{xywh}][{xywhn}]")
                    window_name = str(len(crops))
                    cv2.imshow(window_name, crops[-1])
                # print(result.boxes.cpu().numpy())
                cv2.waitKey(0)
            if save_images: # Save annotated image
                save_file = join(output_dir,f"result{c:03d}.jpg")
                result.save(filename=save_file)  
                print(f"Result Saved: {save_file}")
        
            # Visualize the detection
            if visualize_images:
                res_img = result.plot()  # This plots the detections on the image
                cv2.imshow('Detection Results', res_img)
                cv2.waitKey(delay_ms)

        cv2.destroyAllWindows()
    print("Done")