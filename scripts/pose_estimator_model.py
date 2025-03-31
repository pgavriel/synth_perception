import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np

# Define the dual-input pose estimation model
class PoseEstimationModel(nn.Module):
    def __init__(self):
        super(PoseEstimationModel, self).__init__()

        # Image branch: Convolutional layers to process 96x96x3 images
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Flattened image vector size after convolutions (96x96 -> 12x12 -> 64*12*12)
        self.fc_image = nn.Sequential(
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
        )

        # 1D vector branch: for object id + crop vector (5x1)
        self.fc_vector = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        # Combined branch
        self.fc_combined = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 3 for size, 3 for translation, 4 for quaternion rotation
        )

    def forward(self, image, vector):
        # Image branch
        x_image = self.conv_layers(image)
        x_image = x_image.view(x_image.size(0), -1)
        x_image = self.fc_image(x_image)

        # Vector branch
        x_vector = self.fc_vector(vector)

        # Concatenate both branches
        x = torch.cat((x_image, x_vector), dim=1)
        output = self.fc_combined(x)

        # Split the output: first 6 values (e.g., translation + other params), last 4 for quaternion
        non_quat, quat = output[:, :6], output[:, 6:]
        # Normalize quaternion to ensure it's a unit quaternion
        quat = quat / torch.norm(quat, dim=1, keepdim=True)
        # Concatenate back together
        output = torch.cat((non_quat, quat), dim=1)

        return output

# Dataset class for loading images and labels
class PoseDataLoader(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (96, 96))
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Load corresponding label
        label_path = os.path.join(self.label_dir, self.image_filenames[idx].replace('.png', '.txt'))
        with open(label_path, 'r') as file:
            label = file.readline().strip().split(',')

        object_id = float(label[0])
        crop_vector = np.array(label[1:5], dtype=np.float32)
        vector_input = torch.tensor(np.concatenate(([object_id], crop_vector)), dtype=torch.float32)

        size_vector = np.array(label[5:8], dtype=np.float32) # currently unused 
        # size_vector = (size_vector - np.min(size_vector)) / (np.max(size_vector) - np.min(size_vector)) # Normalize
        # print(f"Size: {size_vector}")
        translate_vector = np.array(label[8:11], dtype=np.float32)
        rotation_vector = np.array(label[11:15], dtype=np.float32)
        target = torch.tensor(np.concatenate((size_vector, translate_vector, rotation_vector)), dtype=torch.float32)

        return image, vector_input, target

# Create data loaders
def get_data_loaders(image_root, label_root, batch_size=16):
    train_dataset = PoseDataLoader(os.path.join(image_root, 'train'), os.path.join(label_root, 'train'))
    val_dataset = PoseDataLoader(os.path.join(image_root, 'val'), os.path.join(label_root, 'val'))
    # print(train_dataset[0])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# Example usage
image_root = '/home/csrobot/synth_perception/data/pose-estimation/test/images'
label_root = '/home/csrobot/synth_perception/data/pose-estimation/test/labels'
train_loader, val_loader = get_data_loaders(image_root, label_root)

model = PoseEstimationModel()
print(model)
