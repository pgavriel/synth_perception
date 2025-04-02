import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib
# print(matplotlib.rcsetup.all_backends)
matplotlib.use('TkAgg')  # or 'Agg' for non-GUI use
import os
from os.path import join
from torch.utils.data import DataLoader
from pose_estimator_model import PoseEstimationModel, PoseDataLoader
import utilities as util

# Training function
def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, output_dir):
    model.to(device)
    train_losses, val_losses = [], []
    size_losses, tran_losses, rot_losses = [], [], []
    size_loss_weight = 1.0
    tran_loss_weight = 1.0
    rot_loss_weight = 5.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_size_loss = 0.0
        running_tran_loss = 0.0 
        running_rot_loss = 0.0

        # if epoch % 5 == 0:
        #     tran_loss_weight = 1.0
        #     rot_loss_weight = 0.0
        # else:
        #     tran_loss_weight = 0.0
        #     rot_loss_weight = 5.0

        for images, vectors, labels in train_loader:
            images, vectors, labels = images.to(device), vectors.to(device), labels.to(device)
            optimizer.zero_grad()
            # print(f"[Img: {images.type} {images.shape}][Vector: {vectors.type} {vectors.shape}]")
            outputs = model(images, vectors)
            size_loss = size_loss_weight * criterion(outputs[:,:3], labels[:,:3])
            tran_loss = tran_loss_weight * criterion(outputs[:,3:6], labels[:,3:6])
            rot_loss = rot_loss_weight * criterion(outputs[:,6:], labels[:,6:])
            loss = size_loss + tran_loss + rot_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_size_loss += size_loss.item()
            running_tran_loss += tran_loss.item()
            running_rot_loss += rot_loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        size_losses.append(running_size_loss / len(train_loader))
        tran_losses.append(running_tran_loss / len(train_loader))
        rot_losses.append(running_rot_loss / len(train_loader))

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, vectors, labels in val_loader:
                images, vectors, labels = images.to(device), vectors.to(device), labels.to(device)
                outputs = model(images, vectors)
                size_loss = size_loss_weight * criterion(outputs[:,:3], labels[:,:3])
                tran_loss = tran_loss_weight * criterion(outputs[:,3:6], labels[:,3:6])
                rot_loss = rot_loss_weight * criterion(outputs[:,6:], labels[:,6:])
                loss = size_loss + tran_loss + rot_loss
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save model checkpoint
        if (epoch+1) % 10 == 0 or epoch+1 == num_epochs:
            torch.save(model.state_dict(), os.path.join(output_dir, f'model_epoch_{epoch+1}.pth'))
            print(f"[Saved Chekpoint] model_epoch_{epoch+1}.pth")

    # Plot training and validation loss
    plt.plot(range(1, num_epochs + 1), size_losses, label='Size Loss')
    plt.plot(range(1, num_epochs + 1), tran_losses, label='Translation Loss')
    plt.plot(range(1, num_epochs + 1), rot_losses, label='Rotation Loss')
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.show()

# Main training script
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    output_root = '/home/csrobot/synth_perception/runs/pose_estimation'
    prefix = "mustard"
    output_dir = util.create_incremental_dir(output_root,prefix)
    print(f"OUTPUT DIRECTORY: {output_dir}")

    data_root = "/home/csrobot/synth_perception/data/pose-estimation/test2"
    train_dataset = PoseDataLoader(join(data_root,"images/train"), join(data_root,"labels/train"))
    val_dataset = PoseDataLoader(join(data_root,"images/val"), join(data_root,"labels/val"))
    # print(train_dataset[0])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = PoseEstimationModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, output_dir)

if __name__ == '__main__':
    main()
