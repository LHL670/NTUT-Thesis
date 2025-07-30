#
# File: train_VIGOR_wicaf_wlum_adapted2d_wDA_wmask_wmixstyle.py (FINAL VERSION)
# Description: Training script with full MixStyle controls.
#

import os
import argparse
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from loss.losses_wicaf_wDA import infoNCELoss
from dataset.datasets_wicaf_wDA_wmask import VIGORDataset
# 引入最終修正版的模型
from model.models_wicaf_wlum_adapted2d_wDA_wmask_wmixstyle import CVM_VIGOR as CVM
from datetime import datetime
import shutil
import yaml
import torch.optim.lr_scheduler
from tqdm import tqdm

# Setup
torch.manual_seed(17)
np.random.seed(0)
currentTime = datetime.now().strftime('%Y%m%d_%H%M%S')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The device is: {device}")

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--area', type=str, help='samearea or crossarea', default='crossarea')
parser.add_argument('--training', choices=('True','False'), default='True')
parser.add_argument('--pos_only', choices=('True','False'), default='True')
parser.add_argument('-l', '--learning_rate', type=float, help='learning rate', default=8e-5)
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=10)
parser.add_argument('-f', '--FoV', type=int, help='field of view', default=360)
parser.add_argument('--ori_noise', type=float, help='noise in orientation prior', default=180.)
parser.add_argument('--epochs', type=int, default=120, help='number of training epochs')
parser.add_argument('--d_model', type=int, default=320, help='Transformer embedding dimension (channels)')
parser.add_argument('--n_layer', type=int, default=2, help='Number of stacked ICFE blocks')
parser.add_argument('--vert_anchors', type=int, default=16, help='Vertical anchors for SFS pooling')
parser.add_argument('--horz_anchors', type=int, default=16, help='Horizontal anchors for SFS pooling')
parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--block_exp', type=int, default=4, help='Expansion factor for MLP')
parser.add_argument('--patience', type=int, default=7, help='Patience for early stopping')
parser.add_argument('--min_delta', type=float, default=0.01, help='Min delta for early stopping')
parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for infoNCELoss')
parser.add_argument('--weight_decay', type=float, default=2e-3, help='Weight decay for AdamW optimizer')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint for resuming training')
parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for decoder conv blocks')
parser.add_argument('--use_mask', type=lambda x: (str(x).lower() == 'true'), default=True, help='Use semantic mask during training and validation')
parser.add_argument('--dataset_root', type=str, default=None, help='Path to dataset')

# MixStyle Arguments
parser.add_argument('--use_mixstyle', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--mixstyle_p', type=float, default=0.3)
parser.add_argument('--mixstyle_alpha', type=float, default=0.1)
parser.add_argument('--mixstyle_mix', type=str, default='random', choices=['random', 'crossdomain'], help="MixStyle strategy")

# Dataset Root

args = parser.parse_args()
dataset_root = args.dataset_root
# dataset_root = '/home/rvl/Documents/hsinling/dataset/VIGOR'
# dataset_root = '/media/lhl/lulu/VIGOR'
# Dataset and DataLoader
print(f"Initializing dataset with use_mask={args.use_mask}")
train_dataset = VIGORDataset(root=dataset_root, split=args.area, train=True, transform=None, pos_only=(args.pos_only=='True'), ori_noise=args.ori_noise, use_mask=args.use_mask)
val_dataset = VIGORDataset(root=dataset_root, split=args.area, train=False, transform=None, pos_only=(args.pos_only=='True'), ori_noise=args.ori_noise, use_mask=args.use_mask)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=device.type == 'cuda')
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=12, pin_memory=device.type == 'cuda')

# Model Initialization
CVM_model = CVM(
    c_in_grd=3, c_in_sat=3, d_model=args.d_model, n_layer=args.n_layer,
    vert_anchors=args.vert_anchors, horz_anchors=args.horz_anchors,
    h=args.num_heads, block_exp=args.block_exp, FoV=args.FoV,
    dropout_rate=args.dropout_rate,
    use_mixstyle=args.use_mixstyle,
    mixstyle_p=args.mixstyle_p,
    mixstyle_alpha=args.mixstyle_alpha,
    mixstyle_mix=args.mixstyle_mix
).to(device)

# Optimizer, Scheduler, Loss
criterion = infoNCELoss
optimizer = torch.optim.AdamW(CVM_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

# Resume and Logging Setup
start_epoch = 0
best_val_distance = float('inf')
early_stopping_counter = 0

if args.resume and os.path.exists(args.resume):
    checkpoint = torch.load(args.resume, map_location=device)
    CVM_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_distance =  float('inf')
    if 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded successfully.")
        except:
            print("Could not load scheduler state. It might be due to a change in scheduler type. Initializing new scheduler.")
    early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
    print(f"Resuming training from {args.resume} starting at epoch {start_epoch}")
else:
    if args.resume:
        print(f"Warning: Resume path '{args.resume}' not found. Starting training from scratch.")
    else:
        print("No resume path provided. Starting training from scratch.")

output_base_dir = 'runs/VIGOR'
run_dir_name = (f"wicaf_wlum_adapted2d_wDA{currentTime}_{args.area}_"
                f"nL{args.n_layer}_T{args.temperature}_wd{args.weight_decay}_"
                f"useMask{args.use_mask}_useMixStyle{args.use_mixstyle}_mix_{args.mixstyle_mix}")
run_dir = os.path.join(output_base_dir, run_dir_name)
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, 'args.yaml'), 'w') as f:
    yaml.dump(vars(args), f, default_flow_style=False)

# Backup scripts
shutil.copy(__file__, os.path.join(run_dir, os.path.basename(__file__)))
shutil.copy('model/models_wicaf_wlum_adapted2d_wDA_wmask_wmixstyle.py', run_dir)
shutil.copy('model/mixstyle.py', run_dir)
shutil.copy('dataset/datasets_wicaf_wDA_wmask.py', run_dir)
shutil.copy('loss/losses_wicaf_wDA.py', run_dir)

log_file_path = os.path.join(run_dir, 'training_log.txt')
with open(log_file_path, 'w') as f:
    f.write("Epoch,Train Loss,Val Loss,Val Mean Distance Error,Val Mediam Distance Error,Learning Rate\n")

print(f"Training results will be saved to: {run_dir}")
print(f"Starting training with label: {run_dir_name}")

# Training Loop
for epoch in range(start_epoch, args.epochs):
    CVM_model.train()
    epoch_total_train_loss = 0.0
    for i, data in enumerate(tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}"), 0):
        grd, sat, gt, mask = data
        grd, sat, mask = grd.to(device), sat.to(device), mask.to(device)
        gt_flattened = gt.flatten(start_dim=1).to(device)

        optimizer.zero_grad()
        logits_flattened, heatmap_output = CVM_model(grd, sat, mask)
        loss = criterion(logits_flattened, gt_flattened, temperature=args.temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(CVM_model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_total_train_loss += loss.item()

    avg_epoch_train_loss = epoch_total_train_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}, Train Loss: {avg_epoch_train_loss:.3f}')

    run_validation = (epoch <  30 and (epoch + 1) % 3 == 0) or (epoch >= 30)
    # run_validation = True
    
    if run_validation:
        print(f"--- Running validation for epoch {epoch + 1} ---")
        CVM_model.eval()
        val_distances_gpu = []
        epoch_total_val_loss = 0.0
        with torch.no_grad():
            for i_val, data_val in enumerate(tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}"), 0):
                grd_val, sat_val, gt_val, mask_val = data_val
                grd_val, sat_val, gt_val, mask_val = grd_val.to(device), sat_val.to(device), gt_val.to(device), mask_val.to(device)
                
                gt_flattened_val = gt_val.flatten(start_dim=1)
                logits_flattened_val, heatmap_output_val = CVM_model(grd_val, sat_val, mask_val)
                val_loss = criterion(logits_flattened_val, gt_flattened_val, temperature=args.temperature)
                epoch_total_val_loss += val_loss.item()
                
                b, c, h, w = gt_val.shape
                gt_max_indices = torch.argmax(gt_val.view(b, -1), dim=1)
                loc_gt_y, loc_gt_x = gt_max_indices // w, gt_max_indices % w
                
                pred_max_indices = torch.argmax(heatmap_output_val.view(b, -1), dim=1)
                loc_pred_y, loc_pred_x = pred_max_indices // w, pred_max_indices % w
                
                pixel_dist = torch.sqrt((loc_gt_y - loc_pred_y)**2 + (loc_gt_x - loc_pred_x)**2)
                val_distances_gpu.append(pixel_dist)

        if val_distances_gpu:
            all_distances_tensor = torch.cat(val_distances_gpu)
            current_val_distance_error = all_distances_tensor.mean().item()
            current_val_mediam_distance_error = all_distances_tensor.median().item()
        else:
            current_val_distance_error = float('inf')
            current_val_mediam_distance_error = float('inf')

        avg_epoch_val_loss = epoch_total_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else float('inf')
        
        print(f'Epoch {epoch + 1}, Val Loss: {avg_epoch_val_loss:.3f}, Val Mean Distance Error: {current_val_distance_error:.2f}, Val Mediam Distance Error:{current_val_mediam_distance_error:.2f}')

        if current_val_distance_error < best_val_distance - args.min_delta:
            print(f"Validation distance error improved from {best_val_distance:.2f} to {current_val_distance_error:.2f}.")
            best_val_distance = current_val_distance_error
            early_stopping_counter = 0
            best_model_path = os.path.join(run_dir, 'best.pt')
            checkpoint_data_best = {
                'epoch': epoch, 'model_state_dict': CVM_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_val_distance': best_val_distance,
                'scheduler_state_dict': scheduler.state_dict(), 'early_stopping_counter': early_stopping_counter,
                'args': vars(args)
            }
            torch.save(checkpoint_data_best, best_model_path)
            print(f"New best model saved to {best_model_path}")
        else:
            early_stopping_counter += 1
            print(f"Validation distance error did not improve for {early_stopping_counter} epoch(s). Early stopping counter: {early_stopping_counter}/{args.patience}")
        
        current_lr = optimizer.param_groups[0]['lr']
        with open(log_file_path, 'a') as f:
            f.write(f"{epoch + 1},{avg_epoch_train_loss:.3f},{avg_epoch_val_loss:.3f},{current_val_distance_error:.2f},{current_val_mediam_distance_error:.2f},{current_lr:.8f}\n")
    
    else:
        print(f"--- Skipping validation for epoch {epoch + 1} ---")
        current_lr = optimizer.param_groups[0]['lr']
        with open(log_file_path, 'a') as f:
            f.write(f"{epoch + 1},{avg_epoch_train_loss:.3f},N/A,N/A,N/A,{current_lr:.8f}\n")

    scheduler.step()
    
    latest_model_path = os.path.join(run_dir, 'latest.pt')
    checkpoint_data_latest = {
        'epoch': epoch, 'model_state_dict': CVM_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 'best_val_distance': best_val_distance,
        'scheduler_state_dict': scheduler.state_dict(), 'early_stopping_counter': early_stopping_counter,
        'args': vars(args)
    }
    torch.save(checkpoint_data_latest, latest_model_path)

    if run_validation and early_stopping_counter >= args.patience:
        print(f"Early stopping triggered! No improvement on validation distance error for {args.patience} epochs.")
        break
    
    print("-" * 50)

print('Finished Training')