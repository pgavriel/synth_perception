from ultralytics import YOLO
from pathlib import Path
import os
from os.path import join, exists
import yaml
import numpy
import json
import csv
from datetime import datetime
import argparse

def convert_label_file(label_path, dest_path, name_to_id):
    lines_out = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        class_name = parts[0]
        if class_name not in name_to_id:
            # Skip labels the model does not have
            continue
        class_id = name_to_id[class_name]
        # Replace class name with class id, leave bbox coords unchanged
        new_line = " ".join([str(class_id)] + parts[1:])
        lines_out.append(new_line)

    # If no boxes remain, output an empty file so YOLO doesn't break
    dest_path.write_text("\n".join(lines_out) if lines_out else "")


# Load a model 
model_folder = "atb1_test2"
parser = argparse.ArgumentParser(description='Provide the model and benchmarks to run.')
parser.add_argument('-m','--model', type=str, default=model_folder,
                    help='Name of the dataset folder to use for training.')
args = parser.parse_args()
model_folder = args.model
print("Loading Model...")
# model_folder = "atb1_test2"
# model_folder = "gear_test_loose"
model_path = join("/home/csrobot/synth_perception/runs/detect",model_folder,"weights/best.pt")
model = YOLO(model_path) # give path to .pt file
name_to_id = {v: k for k, v in model.names.items()}
print(f"Name to ID: {name_to_id}")

benchmark_root = Path("/home/csrobot/Desktop/ATB1_BENCHMARK")
# benchmark_name = "atb1_benchv0"
# benchmark_name = "benchv1A-Dark"
# benchmark_name = "benchv1A-Light"
benchmarks = ["atb1_benchv0","benchv1A-Light","benchv1A-Dark"]
for benchmark_name in benchmarks:
    # Change local directory to the benchmark folder (for saving output results)
    os.chdir(benchmark_root / benchmark_name)
    benchmark_images = benchmark_root / benchmark_name / "images"
    benchmark_labels  = benchmark_root / benchmark_name / "named_labels"
    num_labels = len(list(benchmark_labels.glob("*.txt")))
    # Labels are currently formatted with class names, need to be restructured to use proper class ID
    tmp_labels = benchmark_root / benchmark_name / "labels"
    tmp_labels.mkdir(exist_ok=True)
    num_temp_labels = len(list(tmp_labels.glob("*.txt")))

    # Handle label conversion
    ALWAYS_CONVERT = True
    print(f"Num Labels: {num_labels}\nNum TempLabels: {num_temp_labels}")
    if num_labels != num_temp_labels or ALWAYS_CONVERT:
        for label_file in benchmark_labels.glob("*.txt"):
            convert_label_file(label_file, tmp_labels / label_file.name, name_to_id)
        print("FINISHED CONVERTING LABELS\n")
    else:
        print("SKIPPED LABEL CONVERSION\n")

    # Write dataset Yaml file
    yaml_data = {
            'path': str(benchmark_images.parent),
            'train': None,
            'val': str(benchmark_images),
            'test': str(benchmark_images),
            'names': model.names  # ensures class indexes match model
        }
    yaml_path = benchmark_root / benchmark_name / "data.yaml"
    if not exists(yaml_path) or ALWAYS_CONVERT:
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, sort_keys=False)
        print(f"Data written to {yaml_path}")
    else:
        print("data.yaml already exists, skipping.")

    # Validate Model [ https://docs.ultralytics.com/modes/val ]
    save_json = True
    save_plots = True
    visualize_all = True
    output_name = model_folder # Name of the model being benchmarked
    image_size = 1080
    batch_size = 1 # default: 16
    metrics = model.val(
        data = str(yaml_path),
        conf = 0.001,
        batch = batch_size,
        imgsz = image_size,
        name = output_name,
        save_json = save_json,
        plots = save_plots,
        visualize = visualize_all
    )

    # Print out resulting metrics object
    print(type(metrics))
    for k, v in metrics.__dict__.items():
        if k in ["box","confusion_matrix"]:
            print(f"[ {k} ]: {v.__dict__.keys()}")
            for k2, v2 in v.__dict__.items():
                if type(v2) == numpy.ndarray:
                    print(f"   [ {k2} ]: Numpy Array Shape: {v2.shape}")
                else:
                    print(f"   [ {k2} ]: {v2}")
        else:
            print(f"[ {k} ]: {v}")

    # Print some custom metrics
    print(f"\nOVERALL mAP50-95: {metrics.box.map:.4f}")
    print(f"OVERALL mAP50:    {metrics.box.map50:.4f}")
    print(f"OVERALL mAP75:    {metrics.box.map75:.4f}")
    obj_scores = dict()
    for c, mAP in zip(metrics.confusion_matrix.names,metrics.box.maps):
        obj_scores[c] = mAP
    sorted_items_desc = sorted(obj_scores.items(), key=lambda item: item[1],reverse=True)
    ranked_classes_dict = dict(sorted_items_desc)
    print(f"Sorted by class mAP50-95 (descending):")
    for i, (c, s) in enumerate(ranked_classes_dict.items(), start=1):
        print(f"[{str(i).center(4)}] {str(c).ljust(15)}: {s:.4f}")


    # Assemble some custom metrics to save
    confusion = metrics.confusion_matrix.summary()
    custom_metrics = {
        "model_name": model_folder,
        "overall_mAP_5095": metrics.box.map,
        "overall_mAP_75": metrics.box.map75,
        "overall_mAP_50": metrics.box.map50,
        "ranked_classes": ranked_classes_dict,
        "classes": dict(metrics.names),
        "mAPs": list(metrics.box.maps),
        "confusion": confusion
    }
    metrics_file = join(metrics.save_dir,"metrics.json")
    try:
        with open(metrics_file, 'w') as f:
            json.dump(custom_metrics, f, indent=4)  # indent for pretty-printing
        print(f"Data successfully saved to {metrics_file}")
    except IOError as e:
        print(f"Error saving data to file: {e}")

    # Append to master benchmark results
    results_file = benchmark_root / "benchmark_results.csv"
    # results_file = Path(f"{benchmark_name}_results.csv")
    headers = ["Timestamp", "benchmark_name","model_name", "mAP50-95", "Top 5 classes", "Bottom 5 classes"]
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        benchmark_name,
        model_folder,
        round(metrics.box.map, 4),
        ", ".join(list(ranked_classes_dict.keys())[:5]),
        ", ".join(list(ranked_classes_dict.keys())[-5:])
    ]
    file_exists = results_file.exists()

    with results_file.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)
    print(f"✅ Logged benchmark result for '{model_folder}' to {results_file}")

    print("Done.")
