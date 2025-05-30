import os
from os.path import join
import shutil
import json
import yaml
import random
import cv2
from typing import List
from pathlib import Path
from utilities import make_img_square, canonicalize_quaternion, quat_is_normalized, get_subfolders
from itertools import chain

class UnityToPoseEstimationDataset:
    def __init__(self, input_dirs: List[str], output_dir: str, validation_split = 0.15, crop_size=96, verbose=True):
        """
        Initialize the converter with input directories and an output directory.

        :param input_dirs: List of direcere the YOLO-formatted dataset will be saved.
        """
        self.input_dirs = input_dirs
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Create necessary YOLO subdirectories
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
    
    def get_label_assignments(self):
        print("==== Collecting Label Assignments ====")
        unique_labels = set()

        for i, input_dir in enumerate(self.input_dirs, start=1):
            annotations_path = join(input_dir, 'annotation_definitions.json')

            if not os.path.exists(annotations_path):# or not os.path.exists(images_path):
                print(f"Skipping {input_dir}: Missing required annotation file.")
                continue

            with open(annotations_path, 'r') as f:
                annotations = json.load(f)

            labels = annotations["annotationDefinitions"][0]["spec"]
            lbls = []
            for l in labels:
                unique_labels.add(l['label_name'])
                lbls.append(l["label_name"])

            print(f"[ {i:2}/{len(self.input_dirs):2} ][ {os.path.basename(input_dir).center(15)} ] Labels: {lbls}")


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
        Process each dataset and convert it to the YOLO format.
        """
        # TODO: Add a timer
        total_count_train = 0
        total_count_val = 0
        for i, input_dir in enumerate(self.input_dirs, start=1):
            print(f"\n==== CONVERTING DATASET [{i}/{len(self.input_dirs)}]: {os.path.basename(input_dir)} ====")
            print(f"Path: {input_dir}")
            # Assemble list of all sequence subdirectories
            # ASSUMES: Data subdirectories start with "sequence"
            sequence_dirs = []
            for dirpath, dirnames, filenames in os.walk(input_dir):
                for dirname in dirnames:
                    if dirname.startswith("sequence"):
                        sequence_dirs.append(os.path.join(dirpath, dirname))
            print(f"Found {len(sequence_dirs)} sequence subdirectories...")

            # Optionally shuffle the order of the data, shouldn't matter since it's all randomized
            if self.shuffle:
                sequence_dirs = random.shuffle(sequence_dirs) 
            else:
                sequence_dirs = sorted(sequence_dirs)

            # Reset train/val and label count
            count_train = 0
            count_val = 0
            count_labels = 0
            
            # For each Unity output image...
            # ASSUMES: There is only one step per sequence (gets step0)
            for idx, seq in enumerate(sequence_dirs, start=1):

                # Decide whether data is in train or validation set
                set_choice = "train"
                # Get the validation data from the end of the list
                if idx > len(sequence_dirs) - len(sequence_dirs) * self.validation_split:
                    set_choice = "val"
                    count_val += 1
                else:
                    set_choice = "train"
                    count_train += 1

                # Verify existence of source data and setup output files
                image_source = join(seq,"step0.camera.png")

                if not os.path.exists(image_source):
                    print(f"Image {image_source} not found, skipping.")
                    continue
                image_original = cv2.imread(image_source)
                # Copy image to new location
                # shutil.copy(image_source, image_dest)
                # src_path = "/".join(Path(image_source).parts[-3:])
                # dst_path = "/".join(Path(image_dest).parts[-3:])
                # entry_string = f"[{set_choice.upper().center(5)}][ {src_path.ljust(45)}] -> [ {dst_path.ljust(27)}]"
                entry_string = ""

                label_source = join(seq,"step0.frame_data.json")
                
                if not os.path.exists(label_source):
                    print(f"Image {label_source} not found, skipping.")
                    continue
                
                with open(label_source, 'r') as f:
                    label_json = json.load(f)
                # GET CAMERA INTRINSIC INFORMATION
                camera_dim = label_json["captures"][0]["dimension"]
                camera_matrix = label_json["captures"][0]["matrix"] # Loads as 3x3 flattened to 9x1
                w = int(label_json["captures"][0]["dimension"][0])
                h = int(label_json["captures"][0]["dimension"][1])
                cam_cx = w / 2
                cam_cy = h /2
                cam_fx = camera_matrix[0] * cam_cx # matrix[0][0]
                cam_fy = camera_matrix[4] * cam_cy # matrix[1][1]
                # print(f"WxH: {w} x {h}")

                # GET BOUNDING BOX ANNOTATIONS
                annotations = label_json["captures"][0]["annotations"]
                bb_2d = None
                bb_3d = None
                for ann in annotations:
                    if ann["id"] == "bounding box":
                        if "values" in ann:
                            bb_2d = ann["values"]
                        else:
                            bb_2d = []
                    if ann["id"] == "bounding box 3D":
                        if "values" in ann:
                            bb_3d = ann["values"]
                        else:
                            bb_3d = []
                            # print(f"WARNING: No objects found for {seq}")
                # print(json.dumps(objects, indent=4))
                if bb_2d is None:
                    print(f"ERROR: Didn't find bounding box annotation, skipping...")
                    continue
                if bb_3d is None:
                    print(f"ERROR: Didn't find 3D bounding box annotation, skipping...")
                    continue

                assert len(bb_2d) == len(bb_3d)
                
                # For each object in the scene...
                # ASSUMES: BB2D and BB3D are in the same order
                for bb2, bb3 in zip(bb_2d, bb_3d):
                    assert bb2["instanceId"] == bb3["instanceId"] 
                    # STEP 1: GET 2D BB Crop from original image
                    x1 = int(bb2["origin"][0])
                    y1 = int(bb2["origin"][1])
                    x2 = x1 + int(bb2["dimension"][0])
                    y2 = y1 + int(bb2["dimension"][1])
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
                    category_id = self.yolo_classes[bb3['labelName']]
                    bbcx = bb2["origin"][0] + (bb2["dimension"][0]/2) # 2D BB Center X
                    bbcy = bb2["origin"][1] + (bb2["dimension"][1]/2) # 2D BB Center Y
                    bbw = bb2["dimension"][0] # 2D BB Width
                    bbh = bb2["dimension"][1] # 2D BB Height
                    # Crop Vector Values (As defined in source paper)
                    cvec_u = (bbcx - cam_cx) / cam_fx
                    cvec_v = (bbcy - cam_cy) / cam_fy
                    cvec_w = bbw / cam_fx
                    cvec_h = bbh / cam_fy
                    uvwh = [cvec_u, cvec_v, cvec_w, cvec_h]
                    # print(f" > UVWH: [ {cvec_u} {cvec_v} {cvec_w} {cvec_h} ]")
                    # Get Size and Translation Vector
                    outvec_size = bb3["size"]
                    outvec_translate = bb3["translation"]
                    # Get Rotation Vector Quaternion
                    outvec_rot = bb3["rotation"]
                    # Convert Rotation Quaternion to canonical form (Positive q0)
                    outvec_rot = canonicalize_quaternion(outvec_rot,False)
                    # is_norm = quat_is_normalized(outvec_rot)

                    # Write data to txt file
                    objects = [category_id, uvwh, outvec_size, outvec_translate, outvec_rot]
                    flattened = list(chain.from_iterable(obj if isinstance(obj, list) else [obj] for obj in objects))
                    data_str = ','.join(map(str, flattened))
                    # data_str = f"{category_id},{uvwh},{outvec_size},{outvec_translate},{outvec_rot}"
                    label_dest = join(self.label_dir,set_choice,f"{id_str}.txt")
                    with open(label_dest, 'w') as lf:
                        lf.write(f"{data_str}")
                    # Increment data ID
                    self.current_data_id += 1
                    

                count_labels += len(bb_2d)

                # Print info line for every image processed
                entry_string = f"[{str(idx).center(5)}][ Bounding Boxes: {str(len(bb_2d)).center(10)}]"
                if self.verbose: print(entry_string)
                 
            print(f"[ Train Count: {count_train:4d} ][ Validation Count: {count_val:4d} ]")
            print(f"[ Total Label Count: {(count_labels):5d} ][ Average Label Count: {(count_labels/len(sequence_dirs)):5f} ]")
            print(f"[ Camera Properties ][ {w} x {h} ][ cx:{cam_cx} cy:{cam_cy} ][ fx:{cam_fx:.2f} fy:{cam_fy:.2f} ]\n")
                
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


# Example usage
if __name__ == "__main__":
    unity_root = "/home/csrobot/Unity/SynthData/PoseTesting/gear"
    # unity_datasets = ['engine_fruit' ,'engine_nerve' , 'negative_fruit', 'negative_nerve']
    # unity_datasets = ['mustard_nerve','mustard_fruit','mustard_big']
    # unity_datasets = ['mustard_nerve']
    unity_datasets = get_subfolders("/home/csrobot/Unity/SynthData/PoseTesting/gear")
    convertion_list = [join(unity_root,dataset) for dataset in unity_datasets]

    output_root = "/home/csrobot/synth_perception/data/pose-estimation"
    output_dataset_name = "gear-pose1"
    validation_split = 0.15
    crop_size = 96
    verbose = True

    converter = UnityToPoseEstimationDataset(convertion_list, join(output_root,output_dataset_name),validation_split,crop_size,verbose)
    converter.convert()
    converter.validate_output()