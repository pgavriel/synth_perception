from ultralytics import YOLO, settings
import cv2
from os.path import join, isfile
import os 

def get_image_paths(directory):
    # Define a set of common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    # Get full paths for all files with image extensions in the target directory
    image_paths = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if os.path.splitext(file)[1].lower() in image_extensions
    ]
    return image_paths
# https://docs.ultralytics.com/modes/train/#train-settings 
if __name__ == "__main__":
    # Print Settings
    data_folder = "/home/csrobot/synth_perception/data"
    settings.update({"datasets_dir": data_folder})
    print(settings)

    # Load a model
    model_folder = "train2"
    model_path = join("/home/csrobot/synth_perception/runs/detect",model_folder,"weights/best.pt")
    model = YOLO(model_path)
    model.conf = 0.5
    # Display model information (optional)
    # model.info()
    
    # Run inference with the trained YOLO model
    image_list = ["/home/csrobot/ns-data/engine-18/DSLR/cam1_045_img.jpg",
                  "/home/csrobot/synth_perception/data/test/images/train/00000004.png"]
    image_list = get_image_paths("/home/csrobot/ns-data/engine-18/DSLR/")
    image_list = sorted(image_list)
    # for im in image_list:
    #     print(im)
    # exit()

    c = 1
    for img in image_list:
        results = model(img)
    # results.save_txt("results.txt",True)
        for result in results:    
            # print(f"RESULT\n{result}")
            # result.save_txt(f"results{c}.txt",True)
            # print(result.boxes)  # Print detection boxes
            # result.show()  # Display the annotated image
            result.save(filename=f"testing/result{c:03d}.jpg")  # Save annotated image
            c += 1
    # Visualize the detection
    # res_img = results[0].plot()  # This plots the detections on the image
    # print(f"LEN: {len(results)}")
    # print(results[0].boxes)

        # cv2.imshow('Detection Results', res_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    print("Done")