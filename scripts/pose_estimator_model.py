import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from utilities import canonicalize_quaternion

# Define the dual-input pose estimation model
class PoseEstimationModel(nn.Module):
    def __init__(self):
        super(PoseEstimationModel, self).__init__()

        IMG_BRANCH_OUT_SIZE = 256
        VEC_BRANCH_OUT_SIZE = 256#128

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
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
        )
        
        # Flattened image vector size after convolutions (96x96 -> 12x12 -> 64*12*12)
        self.fc_image = nn.Sequential(
            # nn.Linear(64 * 12 * 12, 128),
            nn.Linear(128*6*6, IMG_BRANCH_OUT_SIZE),
            # nn.Linear(256*3*3, IMG_BRANCH_OUT_SIZE),
            nn.ReLU(),
        )

        # 1D vector branch: for object id + crop vector (5x1)
        self.fc_vector = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, VEC_BRANCH_OUT_SIZE),
            nn.ReLU(),
        )

        # Combined branch
        self.fc_shared = nn.Sequential(
            nn.Linear(VEC_BRANCH_OUT_SIZE + IMG_BRANCH_OUT_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Define separate output branches for each prediciton
        self.fc_size = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        self.fc_translation = nn.Sequential(
            nn.Linear(VEC_BRANCH_OUT_SIZE, 64), # was 128 to match shared branch output size
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        self.fc_rotation = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
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
        x = self.fc_shared(x)

        # Get each prediction head
        size = self.fc_size(x)
        translation = self.fc_translation(x_vector)
        # translation = self.fc_translation(x)
        rotation = self.fc_rotation(x)

        # Normalize quaternion to unit length
        rotation = self._normalize_and_canonicalize_quaternion(rotation)
        
        # Concatenate all outputs: [size, translation, rotation]
        output = torch.cat((size, translation, rotation), dim=1)
        return output
    
    @staticmethod
    def _normalize_and_canonicalize_quaternion(q):
        """
        Ensures q is a unit quaternion and uses a canonical form (q[0] >= 0).
        q: Tensor of shape [batch_size, 4]
        """
        q = F.normalize(q, p=2, dim=1)
        sign = torch.where(q[:, 0:1] < 0, -1.0, 1.0)  # shape: [batch_size, 1]
        return q * sign
    
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
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
