import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from os.path import join

def draw_3d_bounding_box(image, translation, size, rotation,fl=6172):
    # Create 8 corner points of the bounding box
    l, w, h = size
    corners = np.array([
        [-l / 2, -w / 2, -h / 2],
        [l / 2, -w / 2, -h / 2],
        [l / 2, w / 2, -h / 2],
        [-l / 2, w / 2, -h / 2],
        [-l / 2, -w / 2, h / 2],
        [l / 2, -w / 2, h / 2],
        [l / 2, w / 2, h / 2],
        [-l / 2, w / 2, h / 2]
    ])

    # Apply rotation
    rotation_corrected = [rotation[0], -rotation[1], -rotation[2], -rotation[3]]  # Flip x, y, z components

    r = R.from_quat(rotation)  # Rotation as [x, y, z, w]
    # r = R.from_quat(rotation_corrected)  # Rotation as [-x, -y, -z, -w]
    corners_rotated = r.apply(corners)

    # Apply translation
    corners_transformed = corners_rotated + translation

    # Project to 2D assuming simple pinhole camera model
    # For simplicity, use a basic camera intrinsic matrix
    focal_length = fl  # Adjust as needed
    image_center = (image.shape[1] / 2, image.shape[0] / 2)
    intrinsic_matrix = np.array([
        [focal_length, 0, image_center[0]],
        [0, focal_length, image_center[1]],
        [0, 0, 1]
    ])
    print("Intrinsic Matrix:\n",intrinsic_matrix)
    # Convert 3D points to homogeneous coordinates and project
    corners_homogeneous = corners_transformed.T
    print("Corners:\n",corners_homogeneous)
    projected = intrinsic_matrix @ corners_homogeneous
    print("Projected:\n",projected)
    projected /= projected[2]  # Normalize by depth
    print("Projected Normed:\n",projected)

    # Draw the bounding box on the image
    points_2d = projected[:2].T.astype(int)
    print("Points 2d:\n",points_2d)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]
    image = cv2.flip(image, 0)
    for start, end in edges:
        pt1 = tuple(points_2d[start])
        pt2 = tuple(points_2d[end])
        cv2.line(image, pt1, pt2, (0, 255, 0), 1)
    image = cv2.flip(image, 0)
    return image

def visualize_3d_bounding_boxes(json_file, image_file):
    # Load JSON data
    with open(json_file, "r") as f:
        data = json.load(f)
    annotations = data["captures"][0]["annotations"]
    labels_3d = []
    for ann in annotations:
        if ann["id"] == "bounding box 3D":
            if "values" in ann:
                labels_3d = ann["values"]
    # Load image
    image = cv2.imread(image_file)
    if image is None:
        print("Error: Could not load image.")
        return

    # Process each bounding box
    for bbox in labels_3d:
        translation = bbox["translation"]
        size = bbox["size"]
        rotation = bbox["rotation"]  # [x, y, z, w] quaternion
        image = draw_3d_bounding_box(image, translation, size, rotation)

    # Display the image
    cv2.imshow("3D Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
root_dir = "/home/csrobot/Unity/SynthData/EngineTest/solo/sequence.10"
annotations = "step0.frame_data.json"
img = "step0.camera.png"
# visualize_3d_bounding_boxes("path/to/annotation.json", "path/to/image.png")
visualize_3d_bounding_boxes(join(root_dir,annotations), join(root_dir,img))