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
from utilities import get_subfolders
import numpy as np
import glob 
import argparse

class ReplicatorToYOLOConverter:
    def __init__(self, input_dirs: List[str], output_dir: str, validation_split = 0.15, verbose=True):
        """
        Initialize the converter with input directories and an output directory.

        :param input_dirs: List of direcere the YOLO-formatted dataset will be saved.
        """
        self.input_dirs = input_dirs
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # SET BOUNDING BOX MODE
        self.bounding_box_mode = "loose" # "tight" or "loose"
        
        # Create necessary YOLO subdirectories
        self.create_subdirectories()
        # Create YOLO data.yaml file
        self.create_yolo_yaml()

        self.current_data_id = 0
        self.validation_split = validation_split
        self.shuffle = False

        self.objects = {}

        self.verbose = verbose

    def create_subdirectories(self,exist_ok=True):
        # Create necessary YOLO subdirectories
        self.image_dir = join(self.output_dir, 'images')
        self.label_dir = join(self.output_dir, 'labels')
        os.makedirs(self.image_dir, exist_ok=exist_ok)
        os.makedirs(join(self.image_dir,"train"), exist_ok=exist_ok)
        os.makedirs(join(self.image_dir,"val"), exist_ok=exist_ok)
        os.makedirs(self.label_dir, exist_ok=exist_ok)
        os.makedirs(join(self.label_dir,"train"), exist_ok=exist_ok)
        os.makedirs(join(self.label_dir,"val"), exist_ok=exist_ok)
        print("Created subdirectories...")
    
    def get_label_assignments(self, label_file_pattern="bounding_box_2d_{}_labels_{}.json"):
        print("==== Collecting Label Assignments ====")
        unique_labels = set()

        for i, input_dir in enumerate(self.input_dirs, start=1):

            # Gather all annotation files
            label_files = sorted(glob.glob(os.path.join(input_dir, label_file_pattern.format(self.bounding_box_mode,"*"))))
            if len(label_files) == 0:
                    print(f"Skipping {input_dir}: Missing required annotation files.")
                    continue
            # Get labels from each file
            lbls = set()
            for lbl_file in label_files:
                # annotations_path = join(input_dir, 'annotation_definitions.json')
                with open(lbl_file, 'r') as f:
                    labels = json.load(f)
                # if "spec" in annotations["annotationDefinitions"][0]:
                #     labels = annotations["annotationDefinitions"][0]["spec"]
                # else:
                #     labels = []
                # lbls = []
                for l in labels:
                    class_name = labels[l].get("class",None)
                    if class_name is not None:
                        unique_labels.add(class_name)
                        lbls.add(class_name) # make a set

            print(f"[ {i:2}/{len(self.input_dirs):2} ][ {os.path.basename(input_dir).center(15)} ] Labels: {set(lbls)}")


        # TODO: Option to sort labels
        self.yolo_classes = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        print("\nAcquired Class List: ")
        for key, value in self.yolo_classes.items():
                print(f" > {value}: {key}")

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
        Process each dataset and convert it to the YOLO training format.
        """
        image_pattern="rgb_{}.png"
        npy_pattern="bounding_box_2d_{}_{}.npy"
        json_pattern="bounding_box_2d_{}_labels_{}.json"
        OCCLUSION_THRESH = 0.70

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
                
                npy_path = os.path.join(input_dir, npy_pattern.format(self.bounding_box_mode,frame_num))
                json_path = os.path.join(input_dir, json_pattern.format(self.bounding_box_mode,frame_num))

                # Verify each component exists 
                if not os.path.exists(npy_path):
                    print(f"ERROR: Missing .npy file for frame {frame_num}: {npy_path}")
                    continue
                if not os.path.exists(json_path):
                    print(f"ERROR: Missing .json file for frame {frame_num}: {json_path}")
                    continue
                if not os.path.exists(image_path):
                    print(f"ERROR: Image {image_path} not found, skipping.")
                    continue

                # Number ID for data entry
                id_str = f"{self.current_data_id:08d}"

                # Setup output image file
                image_dest = join(self.image_dir,set_choice,f"{id_str}.png")
                # Copy image to new location
                shutil.copy(image_path, image_dest)
                # Create formatted printout
                src_path = "/".join(Path(image_path).parts[-3:])
                dst_path = "/".join(Path(image_dest).parts[-3:])
                entry_string = f"[{set_choice.upper().center(5)}][ {src_path.ljust(45)}] -> [ {dst_path.ljust(27)}]"

                # print(f"Frame: {frame_num}")
                with open(json_path, "r") as f:
                    labels = json.load(f)
                # print(f"Labels: {labels}")

                bboxes = np.load(npy_path)
                # print("Bounding Box Data (.npy):", len(bboxes))
                # ========================================================================

                label_dest = join(self.label_dir,set_choice,f"{id_str}.txt")
                with open(label_dest, 'w') as lf:
                    # Replicator 2D BB Format: (Label ID, x1, y1, x2, y2, occlusion %)
                    entry_string += f"[ Labels: {str(len(bboxes)).center(4)}] "
                    for bb in bboxes:
                        assert len(bb) == 6
                        class_label = labels[str(bb[0])].get("class",None)
                        class_id = self.yolo_classes[class_label]
                        x1 = bb[1]
                        y1 = bb[2]
                        x2 = bb[3]
                        y2 = bb[4]
                        occlusion = bb[5]
                        if occlusion > OCCLUSION_THRESH:
                            excluded_labels += 1
                            # print("Skip occlusion")
                            continue
                        else:
                            count_labels += 1
                        
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        width = abs(x2 - x1) / w
                        height = abs(y2 - y1) / h
                        # Write to YOLO label file
                        lf.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                        # print(f" > {class_id} {x_center} {y_center} {width} {height}")

                        
                # Print info line for every image processed
                if self.verbose: print(entry_string)
                # Increment data ID
                self.current_data_id += 1 
            print(f"[ Train Count: {count_train:4d} ][ Validation Count: {count_val:4d} ][ Excluded Labels: {excluded_labels:5d}][ Average Label Count: {(count_labels/len(image_files)):5f} ]\n")
            total_count_train += count_train
            total_count_val += count_val
            total_excluded += excluded_labels
        print(f"[ Total Train: {total_count_train:5d} ][ Total Val: {total_count_val:5d} ][ Total Excl: {total_excluded:5d}]")


    def validate_output(self):
        """
        Check if the output directory is structured correctly and contains valid data.
        """
        if not os.listdir(self.image_dir):
            raise ValueError("No images found in the output directory.")

        if not os.listdir(self.label_dir):
            raise ValueError("No labels found in the output directory.")

        print("Output directory is valid and contains data.")

## DEV TESTING
def blend_color(unknown):
    """Return a color that linearly blends white (0) to red (1)."""
    red = int(255)
    green = int(255 * (1 - unknown))
    blue = int(255 * (1 - unknown))
    return (blue, green, red)  # BGR format for OpenCV

def load_and_display_bounding_boxes(
    folder_path,
    image_pattern="rgb_{}.png",
    npy_pattern="bounding_box_2d_tight_{}.npy",
    json_pattern="bounding_box_2d_tight_labels_{}.json"
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

        # Draw each bounding box
        for bb in bboxes:
            print(f" > {bb}")
            if len(bb) < 6:
                continue  # Skip malformed entries
            id, x1, y1, x2, y2, occlusion = bb
            color = blend_color(float(occlusion))
            pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(image, pt1, pt2, color, 2)

        # Show image
        cv2.imshow(f"Bounding Boxes (Frame {frame_num})", image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        cv2.destroyAllWindows()

    cv2.destroyAllWindows()

import re
def find_matching_folders(data_root, folder_patterns):
    """
    Recursively search data_root for folders whose names match
    any of the provided regex patterns.

    Args:
        data_root (str): Root directory to search.
        folder_patterns (list[str]): List of regex patterns.

    Returns:
        list[str]: Full paths to matching folders.
    """
    return_paths = []
    for p in folder_patterns:
        found_one = False
        folders = p.split("/")
        pattern = folders.pop(-1)
        compiled = re.compile(pattern)
        folders.insert(0,data_root)
        temp_root = os.path.join(*folders)
        print(f"[{p}]: Temp root: {temp_root} - Pattern: {pattern}")
        for entry in sorted(os.listdir(temp_root)):
            if compiled.fullmatch(entry) or compiled.search(entry):
                found_one = True
                full_path = os.path.join(temp_root,entry)
                return_paths.append(full_path)
                print(f"\t> {full_path}")
        if not found_one:
            print("\tWARNING: NO MATCHES")
    return return_paths

# Example usage
if __name__ == "__main__":
    print("== REPLICATOR DATA -> YOLO DATASET ==")
    parser = argparse.ArgumentParser(description='Process one primary argument and a list of secondary arguments.')

    # Define the first positional argument
    OUTPUT_DATASET_NAME = "test"
    parser.add_argument('dataset_name', type=str, default=OUTPUT_DATASET_NAME,
                        help='The main, required positional argument: output dataset name')

    # Define the argument to collect all remaining arguments as a list
    # nargs=argparse.REMAINDER tells argparse to collect all remaining command-line arguments
    # into a list for this argument.
    parser.add_argument('source_folders', nargs=argparse.REMAINDER, default=[],
                        help='A list of all other subfolders to gather for final dataset.')

    args = parser.parse_args()

    print(f"Dataset Name: {args.dataset_name}")
    print(f"Source Folders: {args.source_folders}")
    data_root = "/home/csrobot/Omniverse/SynthData"
    input_folders = find_matching_folders(data_root, args.source_folders)
    
    
    # Collect a list of full paths to each dataset folder to include
    # replicator_root = "/home/csrobot/Omniverse/SynthData/atb1"
    # replicator_datasets = get_subfolders(replicator_root)
    # replicator_datasets = sorted(replicator_datasets)
    # rep_full_list = [join(replicator_root,dataset) for dataset in replicator_datasets]
    # print(replicator_datasets)

    # # Include negative example data
    # negative_root = "/home/csrobot/Omniverse/SynthData/negative"
    # negative_datasets = get_subfolders(negative_root)
    # neg_full_list = [join(negative_root,dataset) for dataset in negative_datasets]
    
    convertion_list = input_folders
    print(f"Input:\n{convertion_list}")
    
    # for d in convertion_list:
    #     load_and_display_bounding_boxes(d)
    # exit(0)
    output_root = "/home/csrobot/synth_perception/data"
    yolo_dataset_name = args.dataset_name
    validation_split = 0.15

    verbose = False

    converter = ReplicatorToYOLOConverter(convertion_list, join(output_root,yolo_dataset_name),validation_split,verbose)
    # exit(0)
    converter.convert()
    converter.validate_output()
    print(f"Dataset output to: {join(output_root,yolo_dataset_name)}")