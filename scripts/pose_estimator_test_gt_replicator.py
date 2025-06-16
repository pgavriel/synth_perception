from ultralytics import YOLO, settings
import cv2
import torch
import numpy as np
from os.path import join, isfile
import os 
import csv
import json
import glob
import utilities as util
from utilities import make_img_square, get_uvwh, get_image_paths, draw_3d_bounding_box, get_files, canonicalize_quaternion, replicator_extract_3dbb_info, replicator_draw_3d_bounding_box
from pose_estimator_model import PoseEstimationModel
from pose_estimator_train import geodesic_loss, rotation_angle_loss, translation_euclidean_loss, translation_loss
import matplotlib.pyplot as plt

"""
This script is used to validate the post estimation model on ground truth labels, using the ground truth 2D BB as the detection
and using the ground truth 3D BB as a ground truth comparison. 
"""


def log_benchmark_metrics(bench_metrics, output_dir, log_name, log_info):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{log_name}.csv")
    log_exists = os.path.isfile(log_file)

    # Compute metric stats
    metric_stats = {}
    for metric_name, values in bench_metrics.items():
        values = np.array(values)
        metric_stats[f"{metric_name}_mean"] = np.mean(values)
        metric_stats[f"{metric_name}_std"] = np.std(values)

    # Use custom or default timestamp
    timestamp = util.timestamp()

    # Open the file in append mode
    with open(log_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # If the file is new, write headers
        if not log_exists:
            header = ['Timestamp']
            header += list(log_info.keys())
            header += list(metric_stats.keys())
            writer.writerow(header)
            print("Log Created.")

        # Assemble row
        row = [timestamp]
        row += [log_info.get(k, "N/A") for k in log_info]
        row += [f"{metric_stats[k]:.4f}" for k in metric_stats]
        writer.writerow(row)
        print("\nLog Appended.")

def plot_benchmark_metrics(bench_metrics, output_dir, file_name="benchmark_loss_metrics",bench_name="None",show_plot=False, bins=50):
    """
    Plot histograms of benchmark loss metrics.

    Parameters:
    - bench_metrics (dict of str -> list of float): Dictionary containing metric lists.
    - output_dir (str): Directory to save the resulting plot image.
    - show_plot (bool): Whether to display the plot before saving.
    - bins (int): Number of bins to use in the histograms.
    """
    num_metrics = len(bench_metrics)
    cols = 2
    rows = (num_metrics + 1) // cols

    plt.figure(figsize=(10, 4 * rows))
    for i, (metric_name, values) in enumerate(bench_metrics.items()):
        plt.subplot(rows, cols, i + 1)
        plt.hist(values, bins=bins, color='skyblue', edgecolor='black')
        plt.title(metric_name.replace("_", " ").title())
        plt.xlabel("Value")
        plt.ylabel("Frequency")

    plt.suptitle(f"Test Model: {file_name}\nBenchmark Set: {bench_name}", fontsize=14)
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{file_name}.png")
    plt.savefig(output_path)
    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    
    # Load Object Size information - Size will be ignored in benchmark metrics
    object_sizes = util.load_json("object_sizes.json")

    # Create output directory for saving images with drawn labels/predictions
    save_images = True
    if save_images:
        img_output_dir = util.create_incremental_dir("/home/csrobot/Pictures/","bench_best")
        os.makedirs(img_output_dir, exist_ok=True)


    visualize_images = True
    delay_ms = 0 # 0 to wait indefinitely for key input on each image
    
    # Whether to log / plot results
    log_results = False

    verbose = True

    # TODO: Determine programatically from camera GT
    FOCAL_LENGTH = 2199 #4000 #6172

    
    # Text options
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)  # White color
    thickness = 2

    # TODO: Load the experiment_log.csv, to get additional information on the model being tested, like training dataset and hyperparams
    # Load the pose estimator model
    pose_model = PoseEstimationModel()
    # model_folder = "mustard_041"
    model_root = "/home/csrobot/synth_perception/runs/pose_estimation"
    model_folder = "uv_engine_022"
    model_file = "model_epoch_100.pth"
    state_dict = torch.load(join(model_root,model_folder,model_file), weights_only=True)
    pose_model.load_state_dict(state_dict)
    pose_model.eval()
    print('Pose Estimation model loaded successfully!')

    # FILE PATTERNS FOR GROUND TRUTH IMAGES AND LABELS

    # Run inference test
    # Get a list of image paths to test the model on 
    # Should be a direct replicator output dataset with 2D (loose) and 3D bounding box labels
    folder_path = "/home/csrobot/Omniverse/SynthData/benchmarking/engine_single_001"
    # folder_path = "/home/csrobot/Omniverse/SynthData/benchmarking/engine_001"

    image_pattern="rgb_{}.png"
    image_files = sorted(glob.glob(os.path.join(folder_path, image_pattern.format("*"))))
    # Restrict the size of test images (mostly for dev testing)
    limit_test_sample = True
    test_sample_size = 25
    if limit_test_sample:
        image_files = image_files[:test_sample_size+1]
    print(f"Found {len(image_files)} images...")
    
    # Setup metrics
    bench_metrics = {
        "rotation_angle_losses": [],
        "geodesic_losses": [],
        "trans_mae_losses": [],
        "trans_euc_losses": []
    }
    bench_info = {
        "model_name": model_folder,
        "model_file": model_file,
        "bench_dataset": folder_path.split('/')[-1],
        "training_dataset": "Unknown" # TODO: load experiment log csv and get this info
    }
    output_dir  = join(folder_path,"benchmarking")

    # For each test image...
    for c, img_path in enumerate(image_files, start=1):
        # Extract frame number
        frame_num = os.path.splitext(os.path.basename(img_path))[0].split("_")[-1]
        if int(frame_num) == 0: continue
        
        orig_img, frame_data = util.replicator_load_frame_data(folder_path,frame_num)
        img = orig_img.copy()

        # GET CAMERA INTRINSIC INFORMATION
        #TODO: These values are currently hardcoded, but could be extracted from camera params files
        FOCAL_LENGTH = 2199 # Represents focal length X and Y
        w = 1920
        h = 1080
        cam_cx = w / 2
        cam_cy = h /2
        cam_fx = FOCAL_LENGTH * cam_cx # camera matrix[0][0]
        cam_fy = FOCAL_LENGTH * cam_cy # camera matrix[1][1]
    
        losses = {}
        instance_id = 1
        # For every pair of (2D/3D) ground truth bounding boxes found...
        for label_pair in frame_data:
            bb2d = label_pair["2d_label"]
            bb3d = label_pair["3d_label"]
            labels3d = replicator_extract_3dbb_info(bbox=bb3d)[0]
        # for bb2d, bb3d, labels3d in zip(bboxes_2d, bboxes_3d, labels):
            if verbose: print(f"\n===== [ IMAGE {c}][ FRAME {frame_num} ][ DETECTION {instance_id}] ==========================================")
            # VISUALIZE GROUND TRUTHS  ======================
            # Draw the ground truth 2D bounding box
            id, x1, y1, x2, y2, occlusion = bb2d
            pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(img, pt1, pt2, color, 1)
            img = cv2.putText(img, str(instance_id), (x1-5,y1-5), font, font_scale, color, thickness)
            # Then, draw the 3D ground truth bounding box
            translation = labels3d[1]
            rotation = labels3d[2]
            size = labels3d[3]
            rotation = canonicalize_quaternion(rotation,False)
            img = replicator_draw_3d_bounding_box(img, translation, size, rotation,FOCAL_LENGTH,color=(255,255,255),verbose=False)
            
            # RUN MODEL INFERENCE =========================
            # # Extract the info from the ground truth to pass into the Pose Estimator
            xyxy = np.asarray([x1, y1, x2, y2])
            pose_input_vector = get_uvwh(orig_img,id,xyxy,FOCAL_LENGTH,verbose=False)
            if verbose: print(f"Model Input Vector LUVWH: {[float(p) for p in pose_input_vector]}")
            pose_input_vector = torch.tensor(pose_input_vector, dtype=torch.float32).unsqueeze(0) 
            pose_input_crop = make_img_square(orig_img[y1:y2,x1:x2]) # Crop [y1:y2,x1:x2]
            pose_input_crop = torch.tensor(pose_input_crop, dtype=torch.float32).permute(2, 0, 1) / 255.0
            pose_input_crop = pose_input_crop.unsqueeze(0) 
            if verbose: print(f"Model Input Size [Crop]:{pose_input_crop.shape}   [Vector]:{pose_input_vector.shape}")
            # print(f"\nGround Truth 3D Vectors:\n[ SIZE ]{size}\n[ TRAN ]{translation}\n[ ROT  ]{[float(r) for r in rotation]}")

            # # Pass input vectors through the pose estimator
            with torch.no_grad():
                output = pose_model(pose_input_crop, pose_input_vector).numpy()
                # out_size = output[0][0:3] # Get size from json file instead
                out_tran = output[0][3:6]
                out_rot  = output[0][6:]
            out_size = util.get_size_vector("engine",object_sizes)
            # EXPERIMENTAL: RESCALING MODEL TRANSLATION (DOWNSCALED BY 1000x IN TRAINING DATA)
            out_tran *= 1000.0




            # print(f"T: {translation}")
            # print(f"Model Output: \n[ SIZE ]{out_size}\n[ TRAN ]{out_tran}\n[ ROT  ]{out_rot}")
            
            distance = np.linalg.norm(translation - out_tran)
            if verbose:
                # Debug output, show comparison between GT and Model estimate
                print("COMPARISON (Ground Truth vs Model Estimate):")
                print(f"\n[ SIZE  ]\n[  GT   ] {size}\n[ MODEL ] {out_size}")
                print(f"\n[ TRANSLATION ]\n[  GT   ] {translation}\n[ MODEL ] {out_tran}")
                print(f"[ DIFF  ] {np.array([float(f'{ev - gt:.2f}') for ev, gt in zip(out_tran,translation)])}")
                
                print("Euclidean Distance:", distance)
                
                print(f"\n[ ROTATION ]\n[  GT   ] {np.array([float(f'{r:.5f}') for r in rotation])}\n[ MODEL ] {out_rot}")
                rot_diff = np.array([float(f'{ev - gt:.5f}') for ev, gt in zip(out_rot,rotation)])
                print(f"[ DIFF  ] {rot_diff} [ ABS SUM ] = {np.sum(abs(rot_diff))}")

            # # Visualize the pose estimator result
            img = replicator_draw_3d_bounding_box(img, out_tran, out_size, out_rot,FOCAL_LENGTH,color=(0,255,255),verbose=False)
            
            # # Debug print out comparing result to ground truth
            rotation = canonicalize_quaternion(rotation)
            gt_rot_tensor = torch.tensor(rotation, dtype=torch.float32).unsqueeze(0) 
            out_rot = canonicalize_quaternion(out_rot)
            out_rot_tensor = torch.tensor(out_rot, dtype=torch.float32).unsqueeze(0) 
            geo_loss = geodesic_loss(out_rot_tensor,gt_rot_tensor)
            ang_loss = rotation_angle_loss(out_rot_tensor,gt_rot_tensor)

            mae_loss = translation_loss(out_tran,translation,convert_to_tensor=True)
            # euc_loss = translation_euclidean_loss(out_tran,translation)
            
            # print(f"\nGeodesic Loss: {geo_loss:.2f}")
            # print(f"Angular Loss: {ang_loss:.2f}")
            # losses[str(instance_id)] = {"Inst":str(instance_id),"Geo":geo_loss.item(), "Ang":ang_loss.item()}
            
            # Update Metrics
            bench_metrics["rotation_angle_losses"].append(ang_loss)
            bench_metrics["geodesic_losses"].append(geo_loss)
            bench_metrics["trans_mae_losses"].append(mae_loss)
            bench_metrics["trans_euc_losses"].append(distance)

            instance_id += 1 
        # Print Losses per detection
        for k in losses:
            print(losses[k])


        # Show image?
        if visualize_images:
            cv2.imshow("Labels Drawn",img)
            k = cv2.waitKey(delay_ms)
            if k == ord('q'):
                break
        # Save image?
        if save_images:
            save_file = join(img_output_dir,f"result{c:03d}.jpg")
            cv2.imwrite(save_file,img)
            print(f"Result Saved: {save_file}")

       
        # cv2.destroyAllWindows()
    # After all images...
    if log_results:
        log_benchmark_metrics(bench_metrics,
                            output_dir=output_dir, 
                            log_name="benchmark_results",
                            log_info=bench_info)
        plot_benchmark_metrics(bench_metrics, 
                            output_dir=output_dir, 
                            file_name=f"bench_{model_folder}",
                            bench_name=folder_path,
                            show_plot=True)

    print("Done")