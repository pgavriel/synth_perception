import os
import sys
from os.path import join
import shutil
import json
import yaml
import random
import cv2
from typing import List
from pathlib import Path
from utilities import get_subfolders, make_img_square, replicator_extract_3dbb_info, canonicalize_quaternion, get_uvwh
import numpy as np
import glob 
from scipy.spatial.transform import Rotation as R
from itertools import chain

class ReplicatorToPoseEstimationDataset:
    def __init__(self, input_dirs: List[str], output_dir: str, validation_split = 0.15, crop_size=96, verbose=True):
        """
        Initialize the converter with input directories and an output directory.

        :param input_dirs: List of direcere the YOLO-formatted dataset will be saved.
        """
        self.input_dirs = input_dirs
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # SET BOUNDING BOX MODE
        self.bounding_box_mode = "loose" # "tight" or "loose"
        self.occlusion_thresh = 0.50
        self.translation_scale_factor = 1.0

        # Create necessary subdirectories
        self.create_subdirectories()
        # Create YOLO data.yaml file
        self.create_yolo_yaml()

        self.current_data_id = 0
        self.validation_split = validation_split
        self.shuffle = False

        self.objects = {}

        self.crop_size = crop_size
        self.verbose = verbose

    def create_subdirectories(self,exist_ok=True):
        # Create necessary subdirectories
        self.image_dir = join(self.output_dir, 'images')
        self.label_dir = join(self.output_dir, 'labels')
        os.makedirs(self.image_dir, exist_ok=exist_ok)
        os.makedirs(join(self.image_dir,"train"), exist_ok=exist_ok)
        os.makedirs(join(self.image_dir,"val"), exist_ok=exist_ok)
        os.makedirs(self.label_dir, exist_ok=exist_ok)
        os.makedirs(join(self.label_dir,"train"), exist_ok=exist_ok)
        os.makedirs(join(self.label_dir,"val"), exist_ok=exist_ok)
        print("Created subdirectories...\n")
    
    def get_label_assignments(self, label_file_pattern="bounding_box_2d_{}_labels_{}.json"):
        print("==== Collecting Label Assignments ====")
        print(f"Using label file pattern: {label_file_pattern}")
        unique_labels = set()

        for i, input_dir in enumerate(self.input_dirs, start=1):

            # Gather all annotation files
            label_files = sorted(glob.glob(os.path.join(input_dir, label_file_pattern.format(self.bounding_box_mode,"*"))))
            if len(label_files) == 0:
                    print(f"Skipping {input_dir}: Missing required annotation files.")
                    continue
            # Get labels from each file
            for lbl_file in label_files:
                # annotations_path = join(input_dir, 'annotation_definitions.json')
                with open(lbl_file, 'r') as f:
                    labels = json.load(f)
                # if "spec" in annotations["annotationDefinitions"][0]:
                #     labels = annotations["annotationDefinitions"][0]["spec"]
                # else:
                #     labels = []
                lbls = []
                for l in labels:
                    class_name = labels[l].get("class",None)
                    if class_name is not None:
                        unique_labels.add(class_name)
                        lbls.append(class_name) # make a set

            print(f"[ {i:2}/{len(self.input_dirs):2} ][ {os.path.basename(input_dir).center(15)} ] Labels: {set(lbls)}")


        # TODO: Option to sort labels
        self.yolo_classes = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        print("\nAcquired Class List: ")
        for key, value in self.yolo_classes.items():
            print(f" > {value}: {key}")
        print("Labels collected...\n")

    def create_yolo_yaml(self):
        """
        Searches through each Unity dataset to create the dictionary for labels
        """
        self.yolo_classes = dict()
        # Establish class list for all target datasets
        self.get_label_assignments()
 
        yolo_yaml_data = dict()
        yolo_yaml_data["path"] = self.output_dir
        yolo_yaml_data["train"] = "images/train"
        yolo_yaml_data["val"] = "images/val"
        # yolo_yaml_data["test"] = join(self.output_dir,"images/test") # Optional

        # Invert the class dictionary and sort by the numeric keys
        inverted_class_dict = {value: key for key, value in self.yolo_classes.items()}
        sorted_class_dict = dict(sorted(inverted_class_dict.items()))  # Ensure keys are in order
        yolo_yaml_data["names"] = sorted_class_dict

        # Write the YAML file
        output_file = join(self.output_dir,"data.yaml")
        with open(output_file, "w") as file:
            yaml.dump(yolo_yaml_data, file, default_flow_style=False, sort_keys=False)

        print(f"YAML file written to {output_file}")


    def convert(self):
        """
        Process each dataset and convert it to the pose estimation model format.
        """
        # Data processing for pose estimation requires both 2D bb annotations (for cropping), and 3D bb annotations (for transform information)
        image_pattern="rgb_{}.png"
        bb_2d_npy_pattern="bounding_box_2d_{}_{}.npy"
        bb_2d_json_pattern="bounding_box_2d_{}_labels_{}.json"
        bb_3d_npy_pattern="bounding_box_3d_{}.npy"
        bb_3d_json_pattern="bounding_box_3d_labels_{}.json"
        OCCLUSION_THRESH = self.occlusion_thresh

        # TODO: Add a timer
        total_count_train = 0
        total_count_val = 0
        total_excluded = 0

        # For each dataset being included...
        for i, input_dir in enumerate(self.input_dirs, start=1):
            print(f"\n==== CONVERTING DATASET [{i}/{len(self.input_dirs)}]: {os.path.basename(input_dir)} ====")
            print(f"Path: {input_dir}")

            #TODO: Import image size from config json or some other means...
            w, h = (1920, 1080)

            # Reset train/val and label count
            count_train = 0
            count_val = 0
            count_labels = 0
            excluded_labels = 0
            image_files = sorted(glob.glob(os.path.join(input_dir, image_pattern.format("*"))))

            # For each RGB image in the dataset...
            for idx, image_path in enumerate(image_files, start=1):
                # Get frame num and paths for label files
                frame_num = os.path.splitext(os.path.basename(image_path))[0].split("_")[-1]
                # Skip frame 0 which never renders properly for some reason
                # TODO: Fix this in replicator
                if int(frame_num) == 0: continue

                # Decide whether data is in train or validation set
                set_choice = "train"
                # Get the validation data from the end of the list
                if idx > len(image_files) - len(image_files) * self.validation_split:
                    set_choice = "val"
                    count_val += 1
                else:
                    set_choice = "train"
                    count_train += 1
                
                # Get annotation file paths
                bb_2d_npy_path = os.path.join(input_dir, bb_2d_npy_pattern.format(self.bounding_box_mode,frame_num))
                bb_2d_json_path = os.path.join(input_dir, bb_2d_json_pattern.format(self.bounding_box_mode,frame_num))
                bb_3d_npy_path = os.path.join(input_dir, bb_3d_npy_pattern.format(frame_num))
                bb_3d_json_path = os.path.join(input_dir, bb_3d_json_pattern.format(frame_num))

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
                if not os.path.exists(image_path):
                    print(f"ERROR: Image {image_path} not found, skipping.")
                    continue

                # Load RGB Image
                image_original = cv2.imread(image_path)

                # GET CAMERA INTRINSIC INFORMATION
                #TODO: These values are currently hardcoded, but could be extracted from camera params files
                FOCAL_LENGTH = 2199 # Represents focal length X and Y
                # w = 1920
                # h = 1080
                # cam_cx = w / 2
                # cam_cy = h /2
                # cam_fx = FOCAL_LENGTH * cam_cx # camera matrix[0][0]
                # cam_fy = FOCAL_LENGTH * cam_cy # camera matrix[1][1]

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
                # print("Bounding Box Data (.npy):", len(bboxes_2d))
                # ========================================================================

                # For each object in the scene...
                # ASSUMES: BB2D and BB3D are in the same order
                # NOTE: This assumption appears correct by manually inspecting the prim_paths annotation files. Need to do a larger test.
                excluded_occ = 0
                for bb2, bb3 in zip(bboxes_2d, bboxes_3d):
                    # STEP 1: GET 2D BB Crop from original image
                    id_2d, x1, y1, x2, y2, occlusion = bb2
                    xyxy = np.asarray([x1, y1, x2, y2])
                    # If occlusion exceeds threshold, skip this object.
                    if occlusion > OCCLUSION_THRESH:
                        excluded_occ += 1
                        continue
                    # Extract crop, make it square, and reduce it to specified size
                    crop = image_original[y1:y2,x1:x2]
                    crop = make_img_square(crop,self.crop_size)
                     # Number ID for data entry
                    id_str = f"{self.current_data_id:08d}"
                    image_dest = join(self.image_dir,set_choice,f"{id_str}.png")
                    # Save Crop Image
                    cv2.imwrite(image_dest,crop)

                    # STEP 2: Extract BB information and camera intrinsics
                    # FORMAT: LabelID, bbcenterx, bbcentery, bbwidth, bbheight,
                    class_label = class_labels_2d[str(id_2d)].get("class",None)
                    class_id = self.yolo_classes[class_label]
                    # bbcx = (x1 + x2)/2 # 2D BB Center X
                    # bbcy = (y1 + y2)/2 # 2D BB Center Y
                    # bbw = abs(x2 - x1) # 2D BB Width
                    # bbh = abs(y2 - y1) # 2D BB Height
                    # # Crop Vector Values (As defined in source paper)
                    # cvec_u = (bbcx - cam_cx) / cam_fx
                    # cvec_v = (bbcy - cam_cy) / cam_fy
                    # cvec_w = bbw / cam_fx
                    # cvec_h = bbh / cam_fy
                    # uvwh = [cvec_u, cvec_v, cvec_w, cvec_h]
                    luvwh = get_uvwh(image_original, class_id, xyxy, FOCAL_LENGTH,verbose=False)
                    
                    # STEP 3: Get 3D BB Info
                    id_3d, tran, rot, scaled_size = replicator_extract_3dbb_info(bb3,verbose=False)[0]
                    # Apply scaling to GT translation vector
                    tran = tran * self.translation_scale_factor 
                    # Convert Rotation Quaternion to canonical form (Positive q0)
                    rot = canonicalize_quaternion(rot,False)
                    
                    # Write data to txt file
                    # objects = [class_id, uvwh, list(scaled_size), list(tran), list(rot)]
                    objects = [luvwh, list(scaled_size), list(tran), list(rot)]
                    # print(f"Objects: {objects}")
                    flattened = list(chain.from_iterable(obj if isinstance(obj, list) else [obj] for obj in objects))
                    data_str = ','.join(map(str, flattened))
                    # print(f"Flat: {flattened}")
                    label_dest = join(self.label_dir,set_choice,f"{id_str}.txt")
                    with open(label_dest, 'w') as lf:
                        lf.write(f"{data_str}")

                    # Increment data ID
                    self.current_data_id += 1

                count_labels += (len(bboxes_2d)-excluded_occ)
                        
                # Print info line for every image processed
                entry_string = f"[{str(idx).center(5)}][ Bounding Boxes: {str(len(bboxes_2d)-excluded_occ).center(10)}]"
                if self.verbose: print(entry_string)
                
            print(f"[ Train Count: {count_train:4d} ][ Validation Count: {count_val:4d} ]")
            print(f"[ Total Label Count: {(count_labels):5d} ][ Average Label Count: {(count_labels/(len(image_files)-1)):5f} ]")
            # print(f"[ Camera Properties ][ {w} x {h} ][ cx:{cam_cx} cy:{cam_cy} ][ fx:{cam_fx:.2f} fy:{cam_fy:.2f} ]\n")
                
            total_count_train += count_train
            total_count_val += count_val
        print(f"[ Total Train: {total_count_train:5d} ][ Total Val: {total_count_val:5d} ]")


    def validate_output(self):
        """
        Check if the output directory is structured correctly and contains valid data.
        """
        print(f"\nChecking output dir: {self.output_dir}")
        if not os.listdir(self.image_dir):
            raise ValueError("No images found in the output directory.")

        if not os.listdir(self.label_dir):
            raise ValueError("No labels found in the output directory.")

        print("Output directory is valid and contains data.")

## DEV TESTING ====================================================================================================================

from utilities import replicator_draw_3d_bounding_box, replicator_extract_3dbb_info

def blend_color(unknown):
    """Return a color that linearly blends white (0) to red (1)."""
    red = int(255)
    green = int(255 * (1 - unknown))
    blue = int(255 * (1 - unknown))
    return (blue, green, red)  # BGR format for OpenCV

def load_and_display_bounding_boxes(
    folder_path,
    image_pattern="rgb_{}.png",
    npy_pattern="bounding_box_3d_{}.npy",
    json_pattern="bounding_box_3d_labels_{}.json",
    draw_and_show=True
):
    image_files = sorted(glob.glob(os.path.join(folder_path, image_pattern.format("*"))))

    for image_path in image_files:
        frame_num = os.path.splitext(os.path.basename(image_path))[0].split("_")[-1]
        if int(frame_num) == 0: continue
        npy_path = os.path.join(folder_path, npy_pattern.format(frame_num))
        json_path = os.path.join(folder_path, json_pattern.format(frame_num))

        if not os.path.exists(npy_path):
            print(f"ERROR: Missing .npy file for frame {frame_num}: {npy_path}")
            sys.exit(1)
        if not os.path.exists(json_path):
            print(f"ERROR: Missing .json file for frame {frame_num}: {json_path}")
            sys.exit(1)

        # Load data
        image = cv2.imread(image_path)
        if image is None:
            print(f"ERROR: Could not load image: {image_path}")
            sys.exit(1)

        print(f"Frame: {frame_num}")
        with open(json_path, "r") as f:
            labels = json.load(f)
        print(f"Labels: {labels}")

        bboxes = np.load(npy_path)
        print("Bounding Box Data (.npy):", len(bboxes))
        for bb in bboxes:
            semantic_id = bb[0]
            x_min = bb[1]
            y_min = bb[2]
            z_min = bb[3]
            x_max = bb[4]
            y_max = bb[5]
            z_max = bb[6]
            
            xyz_min =np.array([x_min, y_min, z_min])
            xyz_max =np.array([x_max, y_max, z_max])
            transform = bb[7]
            occlusion = bb[8]
            # print(bb)
            # print(f"ID: {semantic_id}")
            # print(f"X:{x_min:.2f}:{x_max:.2f} | Y:{y_min:.2f}:{y_max:.2f} | Z:{z_min:.2f}:{z_max:.2f}")
            # print(f"Transform: {transform}")
            # print(f"Occlusion: {occlusion}")
            img = image.copy()
            labels = replicator_extract_3dbb_info(annotation_file=npy_path,verbose=False)
            FOCAL_LENGTH=2199
            if draw_and_show:
                # Draw each bounding box
                for l in labels:
                    translation = l[1]
                    rotation = l[2]
                    size = l[3]
                    # exit()
                    img = replicator_draw_3d_bounding_box(img, translation, size, rotation,FOCAL_LENGTH,color=(255,255,255),verbose=True)
            
                # Show image
                cv2.imshow(f"Bounding Boxes (Frame {frame_num})", img)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
                cv2.destroyAllWindows()

    cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    # Collect a list of full paths to each dataset folder to include
    replicator_root = "/home/csrobot/Omniverse/SynthData/gear_loose"
    replicator_datasets = get_subfolders(replicator_root)
    # replicator_datasets.remove("neg_001")
    # replicator_datasets.remove("neg_002")
    # replicator_datasets = ["t_test_001"]
    # replicator_datasets = sorted(replicator_datasets)
    # replicator_datasets = ["test_007"]
    rep_full_list = [join(replicator_root,dataset) for dataset in replicator_datasets]
    print(replicator_datasets)

    # Include negative example data
    # negative_root = "/home/csrobot/Omniverse/SynthData/negative"
    # negative_datasets = get_subfolders(negative_root)
    # neg_full_list = [join(negative_root,dataset) for dataset in negative_datasets]
    convertion_list = rep_full_list# + neg_full_list
    print(convertion_list)

    # for d in convertion_list:
    #     load_and_display_bounding_boxes(d)
    # exit(0) # Checkpoint 1 =========

    output_root = "/home/csrobot/synth_perception/data/pose-estimation"
    output_dataset_name = "gear_a1"
    validation_split = 0.15

    crop_size = 96
    params = {
        "focal_length": 2199,
        "w": 1920,
        "h": 1080
    }
    verbose = True

    converter = ReplicatorToPoseEstimationDataset(convertion_list, join(output_root,output_dataset_name),validation_split,crop_size,verbose)
    converter.occlusion_thresh = 0.2
    converter.translation_scale_factor = 1#/1000
    # exit(0) # Checkpoint 2 =======

    converter.convert()
    converter.validate_output() # Checkpoint 3 =======