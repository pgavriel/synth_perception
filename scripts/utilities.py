import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import datetime
import json
import random

def timestamp(format="%y-%m-%d-%H-%M-%S"):
    # Get the current time
    now = datetime.datetime.now()
    # Format the time string 
    time_str = now.strftime(format)
    return time_str

def get_random_color_rgb():
    """Generates a random RGB color tuple (values between 0 and 255)."""
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)

def load_json(file,verbose=True):
    # Load Object Size information
    with open("object_sizes.json", "r") as f:
        object_sizes = json.load(f)
    if verbose:
        print("Loaded JSON:")
        print(json.dumps(object_sizes, indent=4))
    return object_sizes

# Get Object size vector or default if it's not found
def get_size_vector(object_name, size_dict):
    return size_dict.get(object_name, size_dict["default"])


def create_incremental_dir(root, prefix="test", digits=3):
    os.makedirs(root, exist_ok=True)  # Ensure root exists
    i = 1
    while True:
        new_dir = os.path.join(root, f"{prefix}_{i:0{digits}}")
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            return new_dir
        i += 1

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

def get_files(search_dir, recursive=True, search_extensions=[".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]):
    # Validate inputs
    search_dir = Path(search_dir)
    if not search_dir.is_dir():
        raise ValueError(f"Search directory '{search_dir}' does not exist or is not a directory.")

    # Find files
    if recursive:
        found_files = search_dir.rglob("*") #sorted(search_dir.rglob("*"))
    else:
        found_files = search_dir.glob("*") #sorted(search_dir.glob("*"))

    found_files = [f for f in found_files if f.suffix.lower() in search_extensions]

    return found_files

def get_subfolders(root_dir):
    return [name for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))]

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


def get_uvwh(image, label, bb_xyxy, fl=3000, verbose=True):
    '''
    Get input vector for pose model from detection
    '''
    height, width, _ = image.shape
    cam_cx = width / 2
    cam_cy = height /2
    cam_fx = fl * cam_cx # camera_matrix[0] * cam_cx # matrix[0][0]
    cam_fy = fl * cam_cy # camera_matrix[4] * cam_cy # matrix[1][1]

    bbw = bb_xyxy[2] - bb_xyxy[0] # bb["dimension"][0] # 2D BB Width
    bbh = bb_xyxy[3] - bb_xyxy[1] # bb["dimension"][1] # 2D BB Height
    bbcx = bb_xyxy[0] + (bbw/2) # 2D BB Center X
    bbcy = bb_xyxy[1]  + (bbh/2) # 2D BB Center Y
    # Crop Vector Values (As defined in source paper)
    cvec_u = (bbcx - cam_cx) / cam_fx
    cvec_v = (bbcy - cam_cy) / cam_fy
    cvec_w = bbw / cam_fx
    cvec_h = bbh / cam_fy
    label_uvwh = [label, cvec_u, cvec_v, cvec_w, cvec_h]
    if verbose:
        print(f"Model Input Vector LUVWH: {label_uvwh}")
    return label_uvwh
    return np.asarray(label_uvwh)

def quat_is_normalized(quaternion, tol=1e-6):
    norm = np.linalg.norm(quaternion)
    print(f"Q Norm: {norm}")
    is_norm = abs(norm - 1.0) < tol
    print(f"Quat is normalized: {is_norm}")
    return is_norm

def canonicalize_quaternion(q, verbose=True):
    q = np.asarray(q)
    if q[0] < 0:
        if verbose: print(f"CANONICALIZED: [{q}] -> [{-q}]")
        q = -q
    return list(q) 

def reconstruct_transform(translation, rotation_quat, scale):
    rotation_matrix = R.from_quat(rotation_quat).as_matrix()
    rot_scaled = rotation_matrix * scale  # Apply scale to columns
    transform = np.eye(4)
    transform[:3, :3] = rot_scaled
    transform[3, :3] = translation
    return transform.T

def decompose_transform_with_size(transform_matrix):
    """
    Decomposes a 4x4 transformation matrix into translation, rotation (quaternion), and scaled size.
    Assumes the matrix is in row-major order.
    """
    # Extract translation (last row, first 3 cols)
    translation = transform_matrix[3, :3].copy()

    # Extract rotation+scale matrix (3x3 part from top-left)
    rot_scale = transform_matrix[:3, :3]

    # Compute per-axis scale as the norm of each column vector
    scale = np.linalg.norm(rot_scale, axis=0)

    # Normalize the rotation matrix
    rotation_matrix = rot_scale / scale

    # Convert rotation matrix to quaternion
    rotation = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]

    # Apply scale to the original size
    # size_scaled = original_size * scale

    return translation, rotation, scale

def draw_3d_bounding_box(image, translation, size, rotation, fl=6172, color=(255,50,150), flip_y=True, flip_x=False, verbose=False):
    """
    Draws bounding boxes for UNITY PERCEPTION DATA
    """
    # Create 8 corner points of the bounding box
    if size is not None:
        l, w, h = size
    else: # Have some default cube size when we want to ignore model size output
        l, w, h = (0.05,0.05,0.05)
    corners = np.array([
        [-l / 2, -w / 2, -h / 2], # -, -, -
        [l / 2, -w / 2, -h / 2],  # +, -, -
        [l / 2, w / 2, -h / 2],   # +, +, -
        [-l / 2, w / 2, -h / 2],  # -, +, -
        [-l / 2, -w / 2, h / 2],  # -, -, +
        [l / 2, -w / 2, h / 2],   # +, -, +
        [l / 2, w / 2, h / 2],    # +, +, +
        [-l / 2, w / 2, h / 2]    # -, +, +
    ])

    # Apply rotation
    # rotation_corrected = [rotation[0], -rotation[1], -rotation[2], -rotation[3]]  # Flip x, y, z components
    norm = quat_is_normalized(rotation)
    # rotation = canonicalize_quaternion(rotation)
    r = R.from_quat(rotation)  # Rotation as [x, y, z, w]
    print(f"Recieved Rotation: {rotation}")
    # r = R.from_quat(rotation_corrected)  # Rotation as [-x, -y, -z, -w]
    corners_rotated = r.apply(corners)

    # Apply translation
    corners_transformed = corners_rotated + translation
    # Convert from Unity coordinates (Y up) to OpenCV (Y down)
    if flip_y: corners_transformed[:, 1] *= -1 # Flip Y
    if flip_x: corners_transformed[:, 0] *= -1 # Flip X


    # Project to 2D assuming simple pinhole camera model
    # For simplicity, use a basic camera intrinsic matrix
    focal_length = fl  # Adjust as needed
    image_center = (image.shape[1] / 2, image.shape[0] / 2)
    intrinsic_matrix = np.array([
        [focal_length, 0, image_center[0]],
        [0, focal_length, image_center[1]],
        [0, 0, 1]
    ])
    if verbose: print("Intrinsic Matrix:\n",intrinsic_matrix)
    # Convert 3D points to homogeneous coordinates and project
    corners_homogeneous = corners_transformed.T
    if verbose: print("Corners:\n",corners_homogeneous)
    projected = intrinsic_matrix @ corners_homogeneous
    if verbose: print("Projected:\n",projected)
    projected /= projected[2]  # Normalize by depth
    if verbose: print("Projected Normed:\n",projected)

    # Draw the bounding box on the image
    points_2d = projected[:2].T.astype(int)
    if verbose: print("Points 2d:\n",points_2d)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]
    ax_x = (0,1)
    ax_y = (1,2)
    ax_z = (1,5)
    draw_edges = True
    draw_corners = True

    if draw_edges:
        for start, end in edges:
            pt1 = tuple(points_2d[start])
            pt2 = tuple(points_2d[end])
            cv2.line(image, pt1, pt2, color, 2)
            if start == ax_x[0] and end == ax_x[1]:
                cv2.line(image, pt1, pt2, (0,0,250), 2)
            if start == ax_y[0] and end == ax_y[1]:
                cv2.line(image, pt1, pt2, (0,250,0), 2)
            if start == ax_z[0] and end == ax_z[1]:
                cv2.line(image, pt1, pt2, (250,0,0), 2)
    if draw_corners:
        for i,p in enumerate(points_2d,start=0):
            c_val = 255 - (50)*(i%4)
            if i < 4:
                image = cv2.circle(image,p,5,(0,0,c_val),-1)
            else:
                image = cv2.circle(image,p,5,(c_val,0,0),-1)    
                
    return image

def replicator_extract_3dbb_info(bbox=None,annotation_file=None,verbose=False):
    """
    Provided a file path to a "bounding_box_3d_####.npy" annotation file, 
    this function will return a list of 4 vector lists that describe the transformations
    of each 3D bounding box label.
    Returns in the format: [ [object_id, translation, rotation, size], ... ]
    """
    if annotation_file is not None:
        bboxes = np.load(annotation_file)
    else:
        bboxes = [bbox]
    labels = []
    # print(f"Bboxes:\n{bboxes}")
    for bb in bboxes:
        # According to Documentation
        # xyz_min =np.array([bb[1], bb[2], bb[5]])
        # xyz_max =np.array([bb[3], bb[4], bb[6]])
        # What looks right (and works)
        xyz_min =np.array([bb[1], bb[2], bb[3]])
        xyz_max =np.array([bb[4], bb[5], bb[6]])
        transform_matrix = bb[7].T
        # Compute size
        size = xyz_max - xyz_min
        if verbose:
            print(f"Size: {size}")
            print(f"Transform: \n{transform_matrix}")

        # Decompose 4x4 matrix
        tran, rot, scale = decompose_transform_with_size(transform_matrix.T)
        # Apply scale to size vector
        scaled_size = size * scale
        if verbose:
            print("Decomposed (Transposed) :")
            print(f"Translation: {tran}")
            print(f"Rotation: \n{rot}")
            print(f"Scale: {scale}")
            print(f"Scaled Size: {scaled_size}")

        # TESTING: Does a reconstructed 4x4 matrix draw properly? (It does)
        # recon_tf = reconstruct_transform(tran,rot,scale)
        # if verbose:
        #     print(f"Reconstructed TF:\n{recon_tf}")
        # labels.append((size, recon_tf))

        # Append decomposed vectors 
        labels.append([bb[0], tran, rot, scaled_size])

    return labels

def replicator_draw_3d_bounding_box(image, translation, size, rotation, fl=6172, color=(255,50,150), flip_y=True, flip_x=True, verbose=False):
    """
    Draws bounding boxes for REPLICATOR DATA
    """
    # Create 8 corner points of the bounding box
    if size is not None:
        l, w, h = size
    else: # Have some default cube size when we want to ignore model size output
        l, w, h = (0.05,0.05,0.05)
    corners = np.array([
        [-l / 2, -w / 2, -h / 2], # -, -, -
        [l / 2, -w / 2, -h / 2],  # +, -, -
        [l / 2, w / 2, -h / 2],   # +, +, -
        [-l / 2, w / 2, -h / 2],  # -, +, -
        [-l / 2, -w / 2, h / 2],  # -, -, +
        [l / 2, -w / 2, h / 2],   # +, -, +
        [l / 2, w / 2, h / 2],    # +, +, +
        [-l / 2, w / 2, h / 2]    # -, +, +
    ])

    # Apply rotation
    norm = quat_is_normalized(rotation)
    # rotation = canonicalize_quaternion(rotation)
    rotation_fixed = [rotation[0], rotation[1], rotation[2], -rotation[3]]
    r = R.from_quat(rotation_fixed)  # Rotation as [x, y, z, w]
    print(f"Recieved Rotation: {rotation}")
    # r = R.from_quat(rotation_corrected)  # Rotation as [-x, -y, -z, -w]
    corners_rotated = r.apply(corners)

    # Apply translation
    corners_transformed = corners_rotated + translation
    # Convert from Unity coordinates (Y up) to OpenCV (Y down)
    if flip_y: corners_transformed[:, 1] *= -1 # Flip Y
    if flip_x: corners_transformed[:, 0] *= -1 # Flip X


    # Project to 2D assuming simple pinhole camera model
    # For simplicity, use a basic camera intrinsic matrix
    focal_length = fl  # Adjust as needed
    image_center = (image.shape[1] / 2, image.shape[0] / 2)
    intrinsic_matrix = np.array([
        [focal_length, 0, image_center[0]],
        [0, focal_length, image_center[1]],
        [0, 0, 1]
    ])
    if verbose: print("Intrinsic Matrix:\n",intrinsic_matrix)
    # Convert 3D points to homogeneous coordinates and project
    corners_homogeneous = corners_transformed.T
    if verbose: print("Corners:\n",corners_homogeneous)
    projected = intrinsic_matrix @ corners_homogeneous
    if verbose: print("Projected:\n",projected)
    projected /= projected[2]  # Normalize by depth
    if verbose: print("Projected Normed:\n",projected)

    # Draw the bounding box on the image
    points_2d = projected[:2].T.astype(int)
    if verbose: print("Points 2d:\n",points_2d)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]
    ax_x = (0,1)
    ax_y = (1,2)
    ax_z = (1,5)
    draw_edges = True
    draw_corners = True

    if draw_edges:
        for start, end in edges:
            pt1 = tuple(points_2d[start])
            pt2 = tuple(points_2d[end])
            cv2.line(image, pt1, pt2, color, 2)
            if start == ax_x[0] and end == ax_x[1]:
                cv2.line(image, pt1, pt2, (0,0,250), 2)
            if start == ax_y[0] and end == ax_y[1]:
                cv2.line(image, pt1, pt2, (0,250,0), 2)
            if start == ax_z[0] and end == ax_z[1]:
                cv2.line(image, pt1, pt2, (250,0,0), 2)
    if draw_corners:
        for i,p in enumerate(points_2d,start=0):
            c_val = 255 - (50)*(i%4)
            if i < 4:
                image = cv2.circle(image,p,5,(0,0,c_val),-1)
            else:
                image = cv2.circle(image,p,5,(c_val,0,0),-1)    
                
    return image


def capture_frames(output_dir="captured_frames", prefix="frame_", digits=3):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)  # Open the default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    frame_count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Camera Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Save frame on 's' key press
            filename = os.path.join(output_dir, f"{prefix}{frame_count:0{digits}}.png")
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            frame_count += 1
        elif key == ord('q'):  # Quit on 'q' key press
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    save_dir = "/home/csrobot/Pictures/collected"
    capture_frames(save_dir)
