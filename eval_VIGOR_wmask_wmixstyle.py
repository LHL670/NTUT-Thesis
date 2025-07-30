import os
import argparse
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset.datasets_wicaf_wDA_wmask import VIGORDataset
from model.models_wicaf_wlum_adapted2d_wDA_wmask_wmixstyle import CVM_VIGOR
from tqdm import tqdm
import csv
from torch.utils.data import DataLoader
from PIL import ImageFile
import yaml
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- 固定隨機種子 ---
torch.manual_seed(17)
np.random.seed(0)
random.seed(17)

# --- 核心參數設定 ---
RESIZE = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ORIGINAL_IMG_SIZE = 640.0

# --- 兩種解析度計算方式 (維持不變) ---
ORIGINAL_METERS_PER_PIXEL = 72.96 / RESIZE
CITY_ORIGINAL_RESOLUTIONS = {
    'Chicago': 0.106,
    'NewYork': 0.113,
    'SanFrancisco': 0.095,
    'Seattle': 0.101
}

# --- 解析命令列參數 ---
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    default='runs/samearea/pos/27.75/best.pt',
                    help='Path to the trained model checkpoint (e.g., best.pt)')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size for evaluation')
parser.add_argument('--dataset_root', type=str, default='/home/lhl/Documents/VIGOR', help='Path to dataset')

args = parser.parse_args()

model_path = args.model_path
batch_size = args.batch_size

# --- 載入訓練時的參數來初始化模型 ---
run_dir = os.path.dirname(model_path)
args_yaml_path = os.path.join(run_dir, 'args.yaml')

if not os.path.exists(args_yaml_path):
    raise FileNotFoundError(f"訓練參數檔案未找到: {args_yaml_path}. 請確認 model_path 指向正確的模型目錄。")

with open(args_yaml_path, 'r') as f:
    train_args = yaml.safe_load(f)

# 從載入的訓練參數中提取所需參數
area = train_args.get('area', 'samearea')
pos_only = train_args.get('pos_only', 'True')
d_model = train_args.get('d_model', 320)
n_layer = train_args.get('n_layer', 2)
vert_anchors = train_args.get('vert_anchors', 16)
horz_anchors = train_args.get('horz_anchors', 16)
num_heads = train_args.get('num_heads', 8)
block_exp = train_args.get('block_exp', 4)
FoV_from_train = train_args.get('FoV', 360)
use_mask = train_args.get('use_mask', True)
dropout_rate = train_args.get('dropout_rate', 0.3)

use_mixstyle = train_args.get('use_mixstyle', True) 
mixstyle_p = train_args.get('mixstyle_p', 0.5)
mixstyle_alpha = train_args.get('mixstyle_alpha', 0.1)
mixstyle_mix = train_args.get('mixstyle_mix', 'random')

# --- 設定輸出資料夾 ---
currentTime = os.path.basename(run_dir)
output_dir = f'results/{currentTime}_eval_compared'
os.makedirs(output_dir, exist_ok=True)
print(f"評估結果將儲存到: {output_dir}")

# --- Dataset ---
dataset_root = args.dataset_root
# dataset_root = '/home/lhl/Documents/VIGOR' # <-- 請確認此路徑
# dataset_root = '/home/rvl/Documents/hsinling/dataset/VIGOR'

vigor = VIGORDataset(dataset_root, split=area, train=False, pos_only=pos_only, use_mask=use_mask)
dataloader = DataLoader(vigor, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# --- 模型載入與初始化 ---
CVM_model = CVM_VIGOR(
    c_in_grd=3, c_in_sat=3,
    d_model=d_model, n_layer=n_layer,
    vert_anchors=vert_anchors, horz_anchors=horz_anchors,
    h=num_heads, block_exp=block_exp,
    FoV=FoV_from_train,
    dropout_rate=dropout_rate,
    use_mixstyle=use_mixstyle,
    mixstyle_p=mixstyle_p,
    mixstyle_alpha=mixstyle_alpha,
    mixstyle_mix=mixstyle_mix
).to(device)

try:
    checkpoint = torch.load(model_path, map_location=device)
    CVM_model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 'N/A')
    best_val_distance = checkpoint.get('best_val_distance', 'N/A')
    print(f"成功載入模型權重: {model_path} (epoch: {epoch}, best_val_distance: {best_val_distance})")
except Exception as e:
    print(f"載入模型權重失敗: {e}")
    exit()

CVM_model.eval()

# --- 批次預測與誤差計算 ---
errors_original = []
errors_city_specific = []
errors_pixel = [] 
results_for_csv = []
index_offset = 0

for batch in tqdm(dataloader, desc="Evaluating"):
    grd_batch, sat_batch, gt_batch, mask_batch = batch
    grd_batch, sat_batch, mask_batch = grd_batch.to(device), sat_batch.to(device), mask_batch.to(device)

    with torch.no_grad():
        _, heatmap_batch = CVM_model(grd_batch, sat_batch, mask_batch)

    heatmap_batch = heatmap_batch.cpu().numpy()
    gt_batch = gt_batch.cpu().numpy()

    for i in range(len(gt_batch)):
        gt, heatmap = gt_batch[i, 0], heatmap_batch[i, 0]
        loc_gt = np.unravel_index(gt.argmax(), gt.shape)
        loc_pred = np.unravel_index(heatmap.argmax(), heatmap.shape)
        pixel_distance = math.sqrt((loc_pred[0] - loc_gt[0])**2 + (loc_pred[1] - loc_gt[1])**2)

        # 儲存像素誤差
        errors_pixel.append(pixel_distance)

        # 計算方法一：原始固定解析度
        dist_meters_original = pixel_distance * ORIGINAL_METERS_PER_PIXEL
        errors_original.append(dist_meters_original)

        # 計算方法二：城市對應解析度
        global_data_idx = index_offset + i
        sat_list_idx = vigor.label[global_data_idx][0]
        sat_path = vigor.sat_list[sat_list_idx]

        try:
            city_name = sat_path.split('/')[-3]
            city_res = CITY_ORIGINAL_RESOLUTIONS.get(city_name, ORIGINAL_METERS_PER_PIXEL)
            meters_per_pixel_city = city_res * (ORIGINAL_IMG_SIZE / RESIZE)
        except IndexError:
            city_name = 'unknown'
            meters_per_pixel_city = ORIGINAL_METERS_PER_PIXEL

        dist_meters_city_specific = pixel_distance * meters_per_pixel_city
        errors_city_specific.append(dist_meters_city_specific)

        filename = os.path.basename(sat_path)
        results_for_csv.append([filename, f"{pixel_distance:.2f}", f"{dist_meters_original:.2f}", f"{dist_meters_city_specific:.2f}", city_name, str(loc_gt), str(loc_pred)])

    index_offset += len(grd_batch)

# --- 計算詳細統計值 ---
def get_stats(error_array):
    if not error_array: # Handle case with no data
        return {key: 0 for key in ["mean", "median", "std", "min", "max", "q1", "q3"]}
    errors = np.array(error_array)
    return {
        "mean": np.mean(errors), "median": np.median(errors),
        "std": np.std(errors), "min": np.min(errors), "max": np.max(errors),
        "q1": np.percentile(errors, 25), "q3": np.percentile(errors, 75)
    }

stats_original = get_stats(errors_original)
stats_city = get_stats(errors_city_specific)
stats_pixel = get_stats(errors_pixel)

# --- 輸出到 Console ---
print("\n" + "="*90)
print(f"評估完成: {area} area, Positive Only: {pos_only}, Use Mask: {use_mask}")
print(f"MixStyle Enabled: {use_mixstyle}, Strategy: {mixstyle_mix if use_mixstyle else 'N/A'}")
print("="*90)
print(f"{'Metric':<10s} | {'Pixel Error (px)':>20s} | {'Original Calc (m)':>20s} | {'City-Specific Calc (m)':>25s}")
print("-"*90)
for key in ["mean", "median", "std", "q1", "q3", "min", "max"]:
    name = key.capitalize() if key not in ['std', 'q1', 'q3'] else {'std': 'Std Dev', 'q1': 'Q1', 'q3': 'Q3'}[key]
    print(f"{name:<10s} | {stats_pixel[key]:>20.2f} | {stats_original[key]:>20.2f} | {stats_city[key]:>25.2f}")
print("="*90)


# --- 儲存 TXT 和 CSV 結果 ---
summary_path = f"{output_dir}/localization_errors_compared_{area}.txt"
with open(summary_path, 'w') as f:
    f.write(f"Model path: {model_path}\n")
    f.write(f"Area: {area}, Positive Only: {pos_only}, Use Mask: {use_mask}\n")
    f.write(f"MixStyle Enabled: {use_mixstyle}, Strategy: {mixstyle_mix if use_mixstyle else 'N/A'}\n\n")

    f.write("--- STATISTICAL ANALYSIS (Pixel Error) ---\n")
    for key, value in stats_pixel.items():
        f.write(f"{key.capitalize():<10s}: {value:.2f} pixels\n")

    f.write("\n--- STATISTICAL ANALYSIS (Original Calculation, meters) ---\n")
    for key, value in stats_original.items():
        f.write(f"{key.capitalize():<10s}: {value:.2f} meters\n")

    f.write("\n--- STATISTICAL ANALYSIS (City-Specific Calculation, meters) ---\n")
    for key, value in stats_city.items():
        f.write(f"{key.capitalize():<10s}: {value:.2f} meters\n")

csv_path = f"{output_dir}/per_image_results_{area}.csv"
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "Pixel Error", "Error (m) Original", "Error (m) City-Specific", "City", "GT (row, col)", "Pred (row, col)"])
    writer.writerows(results_for_csv)

print(f"詳細統計報告已儲存至: {summary_path}")
print(f"每張圖片的詳細結果已儲存至: {csv_path}")

# --- 畫 CED 曲線 (同一張圖比較) ---
# --- 第一張：Original Calc ---
k_values = np.linspace(0, 25, 250)
if errors_original:
    plt.figure(figsize=(8, 5))
    ced_original = [np.mean(np.array(errors_original) <= k) for k in k_values]
    plt.plot(k_values, ced_original, label=f'Original Calc (Median: {stats_original["median"]:.2f}m)', color='royalblue')
    plt.xlabel('Localization Error Threshold (m)')
    plt.ylabel('Cumulative Ratio')
    plt.title('Cumulative Error Distribution (CED) - Original Calc')
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.tight_layout()
    ced_path_original = f"{output_dir}/CED_Original.png"
    plt.savefig(ced_path_original)
    print(f"CED曲線圖 (Original) 已儲存至: {ced_path_original}")

# --- 第二張：City-Specific Calc ---
if errors_city_specific:
    plt.figure(figsize=(8,5))
    ced_city = [np.mean(np.array(errors_city_specific) <= k) for k in k_values]
    plt.plot(k_values, ced_city, label=f'Median: {stats_city["median"]:.2f}m', color='darkorange')
    plt.xlabel('Localization Error Threshold (m)')
    plt.ylabel('Cumulative Ratio')
    plt.title('Cumulative Error Distribution (CED)')
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.tight_layout()
    ced_path_city = f"{output_dir}/CED_CitySpecific.png"
    plt.savefig(ced_path_city)
    print(f"CED曲線圖 (City-Specific) 已儲存至: {ced_path_city}")


# --- 畫誤差直方圖 ---
def plot_histogram(errors, stats, calc_type, output_dir, unit_label):
    plt.figure(figsize=(8, 5))
    if not errors: return
    
    bin_edges = np.arange(0, math.ceil(max(errors)) + 2, 1) if errors else [0, 1]
    plt.hist(errors, bins=bin_edges, edgecolor='black', alpha=0.75)
    plt.axvline(stats['mean'], color='red', linestyle='dashed', linewidth=2, label=f"Mean: {stats['mean']:.2f}{unit_label}")
    plt.axvline(stats['median'], color='green', linestyle='dashed', linewidth=2, label=f"Median: {stats['median']:.2f}{unit_label}")
    plt.xlabel(f'Localization Error ({unit_label})')
    plt.ylabel('Frequency (Number of Images)')
    plt.title(f'Localization Error Histogram ')
    plt.legend()
    plt.grid(axis='y', linestyle=':')
    plt.tight_layout()
    hist_path = f"{output_dir}/Error_Histogram_{calc_type}.png"
    plt.savefig(hist_path)
    print(f"{calc_type} 計算的誤差直方圖已儲存至: {hist_path}")

if errors_pixel:
    plot_histogram(errors_pixel, stats_pixel, 'Pixel', output_dir, unit_label='px')
if errors_original:
    plot_histogram(errors_original, stats_original, 'Original', output_dir, unit_label='m')
if errors_city_specific:
    plot_histogram(errors_city_specific, stats_city, '', output_dir, unit_label='m')