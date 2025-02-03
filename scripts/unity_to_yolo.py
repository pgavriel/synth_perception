import os
from os.path import join
import shutil
import json
import yaml
import random
import cv2
from typing import List
from pathlib import Path

class UnityToYOLOConverter:
    def __init__(self, input_dirs: List[str], output_dir: str, validation_split = 0.15):
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
    
    def get_label_assignments(self):
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
        print("Acquired Class List: ")
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

            # ASSUMES: There is only one step per sequence (gets step0)
            for idx, seq in enumerate(sequence_dirs, start=1):
                # Decide whether data is in train or validation set
                set_choice = "train"
                # Get the validation data from the end of the list
                if idx > len(sequence_dirs) - len(sequence_dirs) * self.validation_split:
                    set_choice = "val"

                # Number ID for data entry
                id_str = f"{self.current_data_id:08d}"

                # Verify existence of source data and setup output files
                image_source = join(seq,"step0.camera.png")
                image_dest = join(self.image_dir,set_choice,f"{id_str}.png")
                if not os.path.exists(image_source):
                    print(f"Image {image_source} not found, skipping.")
                    continue
                # Copy image to new location
                shutil.copy(image_source, image_dest)
                src_path = "/".join(Path(image_source).parts[-3:])
                dst_path = "/".join(Path(image_dest).parts[-3:])
                entry_string = f"[{set_choice.upper().center(5)}][ {src_path.ljust(45)}] -> [ {dst_path.ljust(27)}]"

                label_source = join(seq,"step0.frame_data.json")
                label_dest = join(self.label_dir,set_choice,f"{id_str}.txt")
                if not os.path.exists(label_source):
                    print(f"Image {label_source} not found, skipping.")
                    continue
                
                with open(label_source, 'r') as f:
                    label_json = json.load(f)
                # Get frame size information
                w = int(label_json["captures"][0]["dimension"][0])
                h = int(label_json["captures"][0]["dimension"][1])
                # print(f"WxH: {w} x {h}")

                annotations = label_json["captures"][0]["annotations"]
                objects = None
                for ann in annotations:
                    if ann["id"] == "bounding box":
                        if "values" in ann:
                            objects = ann["values"]
                        else:
                            objects = []
                            # print(f"WARNING: No objects found for {seq}")
                # print(json.dumps(objects, indent=4))
                if objects is None:
                    print(f"ERROR: Didn't find bounding box annotation, skipping...")
                    continue

                # Process labels
                with open(label_dest, 'w') as lf:
                    # print(f" > Detections Found: {len(objects)}")
                    entry_string += f"[ Labels: {str(len(objects)).center(4)}] "
                    for obj in objects:
                        category_id = self.yolo_classes[obj['labelName']]
                        x = obj['origin'][0]
                        y = obj['origin'][1]
                        width = obj['dimension'][0]
                        height = obj['dimension'][1]
                        bbox = [x, y, width, height]
                        # Normalize bounding box
                        # img = cv2.imread(image_source)
                        # h, w, _ = img.shape
                        x_center = (bbox[0] + bbox[2] / 2) / w
                        y_center = (bbox[1] + bbox[3] / 2) / h
                        width = bbox[2] / w
                        height = bbox[3] / h

                        # Write to YOLO label file
                        lf.write(f"{category_id} {x_center} {y_center} {width} {height}\n")
                        # print(f" > {category_id} {x_center} {y_center} {width} {height}")
                
                print(entry_string)
                # Increment data ID
                self.current_data_id += 1 


    def validate_output(self):
        """
        Check if the output directory is structured correctly and contains valid data.
        """
        if not os.listdir(self.image_dir):
            raise ValueError("No images found in the output directory.")

        if not os.listdir(self.label_dir):
            raise ValueError("No labels found in the output directory.")

        print("Output directory is valid and contains data.")

# Example usage
if __name__ == "__main__":
    unity_root = "/home/csrobot/Unity/SynthData/EngineTest"
    unity_datasets = ['solo_15', 'negative_1']
    convertion_list = [join(unity_root,dataset) for dataset in unity_datasets]

    yolo_root = "/home/csrobot/synth_perception/data"
    yolo_dataset_name = "test"
    validation_split = 0.15

    converter = UnityToYOLOConverter(convertion_list, join(yolo_root,yolo_dataset_name),validation_split)
    converter.convert()
    converter.validate_output()