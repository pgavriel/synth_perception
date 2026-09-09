from ultralytics import YOLO
import os
import csv
from os.path import join

# WARNING: Currently a very hacky script for running model benchmarks. Should be refined in the future.

# NEED TO MANUALLY MODIFY MANY VALUES FOR MODELS/BENCHMARKING SCENES
# TODO: IMPLEMENT ARGUMENTS

model_root = "/home/csrobot/synth_perception/runs/detect"

# all_models = ["ex2-moad", "ex2-kiri", "ex2-cadsolidcol", "ex2-cadrandcol", "ex2-cadrandtex"]
all_models = ["ex2-kiri-randmat"]

validation_scenes = ["ex2_006_pose-a", "ex2_009_pose-a", "ex2_024_pose-a", "ex2_026_pose-a"]
# validation_scenes = ["ex2_006_pose-a"]


# ---------------------------------------------------------------------------
# CSV output — one row per (model, scene) combination
# ---------------------------------------------------------------------------
# Columns:
#   model, scene
#   map50, map50_95, precision, recall, f1    ← overall
#   <class>_ap50, <class>_p, <class>_r        ← per class (4 classes)
#
# The CSV is appended so multiple runs accumulate in the same file.
# Change the path here if you want results elsewhere.
CSV_PATH = "/home/csrobot/yolo_results/benchmark_results.csv"
 
# Class names in the order your model was trained — must match data.yaml
CLASS_NAMES = ["conn_wp", "gear_large", "nut_m16", "sprocket_large"]
 
# Build header
per_class_cols = []
for cls in CLASS_NAMES:
    per_class_cols += [f"{cls}_ap50", f"{cls}_p", f"{cls}_r"]
 
HEADER = (
    ["model", "scene",
     "map50", "map50_95", "precision", "recall", "f1"]
    + per_class_cols
)
 
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
write_header = not os.path.isfile(CSV_PATH)
 


for model_name in all_models:
    model_path = join(model_root,model_name,"weights/best.pt")
    model = YOLO(model_path)

    
    for scene in validation_scenes:
        results = model.val(
            data       = f"/home/csrobot/yolo_datasets/{scene}/data.yaml",
            split      = "test",
            imgsz      = 640,          # match your images_4 resolution width
            batch      = 8,             # reduce if you hit VRAM limits
            iou        = 0.6,           # NMS IoU threshold
            save_json  = False,          # saves COCO-format predictions JSON
            save_txt   = True,          # saves per-image prediction .txt files
            plots      = True,          # saves the curve plots above
            project    = "/home/csrobot/yolo_results",   # override default runs/ location
            name       = f"{model_name}-{scene}",            # run subdirectory name
            verbose    = True,          # per-class table printed to terminal
        )

        box = results.box
    
        # Overall metrics — scalar values
        map50     = float(box.map50)
        map50_95  = float(box.map)
        precision = float(box.mp)     # mean precision across classes
        recall    = float(box.mr)     # mean recall across classes
        # F1 is not directly exposed as a scalar — compute from mean p/r
        f1        = (2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0 else 0.0)
    
        print(f"\n{scene}: mAP50={map50:.3f}  mAP50-95={map50_95:.3f}  "
            f"P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}")
    
        # Per-class metrics
        # box.ap50  → per-class AP@50  (list, one per class in model order)
        # box.p     → per-class precision
        # box.r     → per-class recall
        # box.names → {id: name} dict
        model_class_names = results.names   # {0: "conn_wp", 1: "gear_large", ...}
    
        # Build a lookup keyed by class name so column order matches CLASS_NAMES
        # even if the model's internal class ordering differs
        per_class = {}
        for cls_id, cls_name in model_class_names.items():
            per_class[cls_name] = {
                "ap50": float(box.ap50[cls_id]) if cls_id < len(box.ap50) else 0.0,
                "p":    float(box.p[cls_id])    if cls_id < len(box.p)    else 0.0,
                "r":    float(box.r[cls_id])    if cls_id < len(box.r)    else 0.0,
            }
    
        per_class_values = []
        for cls in CLASS_NAMES:
            if cls in per_class:
                per_class_values += [
                    f"{per_class[cls]['ap50']:.4f}",
                    f"{per_class[cls]['p']:.4f}",
                    f"{per_class[cls]['r']:.4f}",
                ]
            else:
                # Class not present in this model — fill with empty
                print(f"  [WARN] class '{cls}' not found in model outputs")
                per_class_values += ["", "", ""]
    
        # Append row to CSV
        row = (
            [model_name, scene,
            f"{map50:.4f}", f"{map50_95:.4f}",
            f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"]
            + per_class_values
        )
    
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(HEADER)
                write_header = False   # only write header once even across scenes
            writer.writerow(row)
    
        print(f"  Row appended → {CSV_PATH}")

        print(f"{scene}: mAP50={results.box.map50:.3f}  mAP50-95={results.box.map:.3f}")