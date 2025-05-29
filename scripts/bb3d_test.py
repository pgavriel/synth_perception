import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from os.path import join
from utilities import draw_3d_bounding_box, replicator_draw_3d_bounding_box,decompose_transform_with_size,reconstruct_transform,replicator_extract_3dbb_info

def visualize_3d_bounding_boxes(annotation_file, image_file):
    # Load image
    image = cv2.imread(image_file)
    if image is None:
        print("Error: Could not load image.")
        return
    
    # Load Annotations (Replicator)
    labels = replicator_extract_3dbb_info(annotation_file=annotation_file,verbose=True)
    
    focal_length = 2199 # Based on camera configuration in replicator
    print(f"Focal Length: {focal_length}")

    fl_inc = 10

    # Process each bounding box
    running = True
    while running: 
        img = image.copy()
        for l in labels:
            # size = l[0]
            # tfm = l[1]
            # size = np.array([0.5,0.5,0.5])
            translation = l[1]
            rotation = l[2]
            size = l[3]
            # exit()
            img = replicator_draw_3d_bounding_box(img, translation, size, rotation,focal_length,color=(255,255,255),verbose=True)
            # img = replicator_draw_3d_bounding_box(img, size, tfm, focal_length,color=(255,255,255),verbose=True)
        cv2.imshow("3D Bounding Boxes", img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            running = False
        elif key == ord('7'): # -FL
            focal_length = focal_length - fl_inc
        elif key == ord('8'): # +FL
            focal_length = focal_length + fl_inc
            print(f"FL:{focal_length}")
    # Display the image
    # cv2.imshow("3D Bounding Boxes", image)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    root_dir = "/home/csrobot/Omniverse/SynthData/dev/stage5_3dpose/test_006"
    annotations = "bounding_box_3d_0002.npy"
    img = "rgb_0002.png"
    # visualize_3d_bounding_boxes("path/to/annotation.json", "path/to/image.png")
    visualize_3d_bounding_boxes(join(root_dir,annotations), join(root_dir,img))