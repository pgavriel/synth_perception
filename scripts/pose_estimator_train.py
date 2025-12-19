import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib
# print(matplotlib.rcsetup.all_backends)
matplotlib.use('TkAgg')  # or 'Agg' for non-GUI use
import os
import traceback
import time
from datetime import timedelta
import csv
from os.path import join
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import ReduceLROnPlateau
from pose_estimator_model import PoseEstimationModel, PoseDataLoader
import utilities as util
import math 

def get_loss_weights(epoch, total_epochs=100, num_cycles=10):
    rot_scale = 10.0
    # Calculate the progress through the total number of cycles (in radians)
    phase = 2 * math.pi * num_cycles * epoch / total_epochs
    # Cosine oscillates from 1 to -1, so shift and scale to get from 1 to 0 to 1
    w_translation = (math.cos(phase) + 1) / 2  # Goes from 1 → 0 → 1
    w_rotation = (1 - w_translation  ) * rot_scale           # Goes from 0 → rot_scale → 0
    return {"t":w_translation, "r":w_rotation}

def compute_loss(output, target, loss_weights={"t":1.0,"r":100.0}, epoch=0):
    size_pred = output[:, 0:3]
    trans_pred = output[:, 3:6]
    quat_pred = output[:, 6:10]

    size_gt = target[:, 0:3]
    trans_gt = target[:, 3:6]
    quat_gt = target[:, 6:10]

    # size_weight = 1.0

    trans_weight = loss_weights["t"]
    rot_weight = loss_weights["r"]

   

    # size_loss = criterion(size_pred, size_gt) * size_weight
    size_loss = 0.0 # We are now ignoring size during training

    # trans_loss = criterion(trans_pred, trans_gt)
    trans_loss = translation_loss(trans_pred, trans_gt) * trans_weight
    # trans_loss2 = translation_euclidean_loss(trans_pred,trans_gt) * trans_weight
    quat_geo_loss = geodesic_loss(quat_pred, quat_gt) * rot_weight
    # quat_angle_loss = rotation_angle_loss(quat_pred, quat_gt) * rot_weight

    total_loss = size_loss + trans_loss + quat_geo_loss #+ trans_loss2 #+ quat_angle_loss

    return total_loss, {
        'size_loss': size_loss,#.item(),
        'trans_loss': trans_loss.item(),
        'quat_geo_loss': quat_geo_loss.item()
        #'quat_angle_loss': quat_angle_loss.item()
    }

def translation_loss(t_pred, t_gt, weights=(1.0, 1.0, 0.1), convert_to_tensor=False):
    if convert_to_tensor: # For benchmarking ONLY, breaks training process
        t_pred = torch.tensor(t_pred, dtype=torch.float32)
        t_gt = torch.tensor(t_gt, dtype=torch.float32)
    loss = F.l1_loss(t_pred, t_gt) # L1 Loss (Mean Absolute Error)    
    return loss 
    # loss = sum(w * F.l1_loss(tp, tg) for tp, tg, w in zip(t_pred, t_gt, weights))
    # return loss

def translation_euclidean_loss(t_pred, t_gt, convert_to_tensor=False):  
    if convert_to_tensor: # For benchmarking ONLY, breaks training process
        t_pred = torch.tensor(t_pred, dtype=torch.float32)
        t_gt = torch.tensor(t_gt, dtype=torch.float32)
    loss = torch.norm(t_pred - t_gt, dim=1).mean() # L2 Loss (Euclidean Distance)
    return loss 

def geodesic_loss(q_pred, q_gt, eps=1e-8):
    verbose = False
    # Normalize both first, just in case
    q_pred = q_pred / (q_pred.norm(dim=-1, keepdim=True) + eps)
    q_gt = q_gt / (q_gt.norm(dim=-1, keepdim=True) + eps)
    if verbose:
        print(f"Pred:\t{q_pred}")
        print(f"GT:\t{q_pred}")
    dot_product = torch.sum(q_pred * q_gt, dim=-1)
    loss = 1.0 - torch.abs(dot_product)
    if verbose:
        print(f"DOT:\t{dot_product}")
        print(f"Loss:\t{loss}")
    return loss.mean()

def rotation_angle_loss(q_pred, q_gt, eps=1e-8):
    verbose = True
    q_pred = q_pred / (q_pred.norm(dim=-1, keepdim=True) + eps)
    q_gt = q_gt / (q_gt.norm(dim=-1, keepdim=True) + eps)

    dot_product = torch.sum(q_pred * q_gt, dim=-1).clamp(-1.0+eps, 1.0-eps)
    angle = 2.0 * torch.acos(torch.abs(dot_product))
    if verbose:
        print(f"Ang DOT:\t{dot_product}")
        print(f"Ang :\t{angle}")
    # return (angle ** 2).mean()
    return (angle).mean()

def rotation_euc_loss(q_pred, q_gt, eps=1e-8):
    verbose = True
    q_pred = q_pred / (q_pred.norm(dim=-1, keepdim=True) + eps)
    q_gt = q_gt / (q_gt.norm(dim=-1, keepdim=True) + eps)

    loss = torch.norm(q_pred - q_gt, dim=1).mean() # L2 Loss (Euclidean Distance)
    if verbose:
        print(f"Ang EUC:\t{loss}")
    return loss 

# Training function
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, output_dir):
    model.to(device)
    train_losses, val_losses = [], []
    size_losses, tran_losses, rot_losses = [], [], []
    rot_ang_losses = []
    val_size_losses, val_tran_losses, val_rot_losses = [], [], []
    val_rot_ang_losses = []
    print_n = 5
    # size_loss_weight = 1.0
    # tran_loss_weight = 1.0
    # rot_geodesic_loss_weight = 5.0
    # rot_angle_loss_weight = 5.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_size_loss = 0.0
        running_tran_loss = 0.0 
        running_rot_loss = 0.0
        running_rot_ang_loss = 0.0

        # if epoch % 5 == 0:
        #     tran_loss_weight = 1.0
        #     rot_loss_weight = 0.0
        # else:
        #     tran_loss_weight = 0.0
        #     rot_loss_weight = 5.0
        n = 1

        # weights = get_loss_weights(epoch)
        # print(f"Weights: {weights}")
        for images, vectors, labels in train_loader:
            images, vectors, labels = images.to(device), vectors.to(device), labels.to(device)
            optimizer.zero_grad()
            # print(f"[Img: {images.type} {images.shape}][Vector: {vectors.type} {vectors.shape}]")
            outputs = model(images, vectors)
            # size_loss = size_loss_weight * criterion(outputs[:,:3], labels[:,:3])
            # tran_loss = tran_loss_weight * criterion(outputs[:,3:6], labels[:,3:6])
            # rot_loss = rot_geodesic_loss_weight * criterion(outputs[:,6:], labels[:,6:])
            # loss = size_loss + tran_loss + rot_loss
            loss, loss_dict = compute_loss(outputs,labels)#,weights,epoch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_size_loss += loss_dict['size_loss']
            running_tran_loss += loss_dict['trans_loss']
            running_rot_loss += loss_dict['quat_geo_loss']
            #running_rot_ang_loss += loss_dict['quat_angle_loss']
            # if n <= print_n:
            #     print(vectors)
            # n += 1

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        size_losses.append(running_size_loss / len(train_loader))
        tran_losses.append(running_tran_loss / len(train_loader))
        rot_losses.append(running_rot_loss / len(train_loader))
        # rot_ang_losses.append(running_rot_ang_loss / len(train_loader))

        # Validation phase
        model.eval()
        val_loss = 0.0
        running_loss = 0.0
        running_size_loss = 0.0
        running_tran_loss = 0.0 
        running_rot_loss = 0.0
        running_rot_ang_loss = 0.0
        with torch.no_grad():
            for images, vectors, labels in val_loader:
                images, vectors, labels = images.to(device), vectors.to(device), labels.to(device)
                outputs = model(images, vectors)
                # size_loss = size_loss_weight * criterion(outputs[:,:3], labels[:,:3])
                # tran_loss = tran_loss_weight * criterion(outputs[:,3:6], labels[:,3:6])
                # rot_loss = rot_geodesic_loss_weight * criterion(outputs[:,6:], labels[:,6:])
                # loss = size_loss + tran_loss + rot_loss
                loss, loss_dict = compute_loss(outputs,labels)#,{"t":1.0,"r":1.0},epoch)
                val_loss += loss.item()
                running_size_loss += loss_dict['size_loss']
                running_tran_loss += loss_dict['trans_loss']
                running_rot_loss += loss_dict['quat_geo_loss']
                #running_rot_ang_loss += loss_dict['quat_angle_loss']

        val_epoch_loss = val_loss / len(val_loader)
        val_losses.append(val_epoch_loss)
        val_size_losses.append(running_size_loss / len(val_loader))
        avg_tran_loss = running_tran_loss / len(val_loader)
        val_tran_losses.append(avg_tran_loss)
        avg_rot_loss = running_rot_loss / len(val_loader)
        val_rot_losses.append(avg_rot_loss)
        #val_rot_ang_losses.append(running_rot_ang_loss / len(val_loader))

        # UPDATE SCHEDULER 
        scheduler.step(val_epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], ",
              f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, ",
              f"LR: {scheduler.get_last_lr()}, ",
              f"[TranLoss: {avg_tran_loss:.3f}][RotLoss: {avg_rot_loss:.3f}]")

        # Save model checkpoint
        if (epoch+1) % 10 == 0 or epoch+1 == num_epochs:
            torch.save(model.state_dict(), os.path.join(output_dir, f'model_epoch_{epoch+1}.pth'))
            print(f"[Saved Chekpoint] model_epoch_{epoch+1}.pth")

    # Plot1: Training and validation loss
    plt.figure(figsize=(20, 6))
    plt.plot(range(1, num_epochs + 1), size_losses, label='Size Loss')
    plt.plot(range(1, num_epochs + 1), tran_losses, label='Translation Loss')
    plt.plot(range(1, num_epochs + 1), rot_losses, label='Rotation Geodesic Loss')
    # plt.plot(range(1, num_epochs + 1), rot_ang_losses, label='Rotation Angular Loss')
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(output_dir, 'training_loss_plot.png'))
    plt.show()

    # Plot1: Training and validation loss
    plt.figure(figsize=(20, 6))
    plt.plot(range(1, num_epochs + 1), val_size_losses, label='Size Loss')
    plt.plot(range(1, num_epochs + 1), val_tran_losses, label='Translation Loss')
    plt.plot(range(1, num_epochs + 1), val_rot_losses, label='Rotation Geodesic Loss')
    # plt.plot(range(1, num_epochs + 1), val_rot_ang_losses, label='Rotation Angular Loss')
    # plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Detailed Validation Loss')
    plt.savefig(os.path.join(output_dir, 'validation_loss_plot.png'))
    plt.show()

    return train_losses[-1], val_losses[-1]

# Main training script
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {}
    config["batch_size"] = 64
    config["epochs"] = 100
    config["learning_rate"] = 0.001
    output_root = '/home/csrobot/synth_perception/runs/pose_estimation'
    prefix = "gear_a"#"gear"
    output_dir = util.create_incremental_dir(output_root,prefix)
    print(f"OUTPUT DIRECTORY: {output_dir}")

    # LOAD TRAINING DATA
    config["training_set"] = "gear_a1"
    data_root = join("/home/csrobot/synth_perception/data/pose-estimation/",config["training_set"])
    train_dataset = PoseDataLoader(join(data_root,"images/train"), join(data_root,"labels/train"))
    val_dataset = PoseDataLoader(join(data_root,"images/val"), join(data_root,"labels/val"))
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # CREATE MODEL 
    model = PoseEstimationModel()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    

    # LOGGING
    write_log = True
    log_file = join(output_root,"experiment_log.csv")
    experiment_name = os.path.basename(output_dir)
    note = ""
    log_info = {"note": "", "error": ""}
    train_loss, val_loss = None, None
    start_time = time.time()

    # TRAIN AND LOG
    try:
        if write_log:
            log_info["note"] = input("Enter test note: ")

        # TRAIN MODEL
        train_loss, val_loss = train(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            device, config["epochs"], output_dir
        )

    except BaseException as e:
        print("\nTraining interrupted or failed:")
        traceback.print_exc()
        log_info["note"] += " [Training Interrupted or Failed]"
        log_info["error"] = f"{type(e).__name__}: {str(e)}"
    finally:
        if write_log:
            # Check if the file exists
            log_exists = os.path.isfile(log_file)

            # Open the file in append mode
            with open(log_file, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write headers if the file is new
                if not log_exists:
                    writer.writerow([
                        'Timestamp', 'Experiment', 'Note', 'Config',
                        'Last Train Loss', 'Last Val Loss', 'Training Time', 'Error'
                    ])
                    print("Log Created.")

                timestamp = util.timestamp()
                elapsed = str(timedelta(seconds=round(time.time() - start_time)))

                writer.writerow([
                    timestamp,
                    experiment_name,
                    log_info["note"],
                    config,
                    train_loss if train_loss is not None else "N/A",
                    val_loss if val_loss is not None else "N/A",
                    elapsed,
                    log_info["error"] 
                ])
                print(f"New Model: {output_dir}")
                print("\nLog Appended.")

if __name__ == '__main__':
    main()
