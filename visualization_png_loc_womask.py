import os
import argparse
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
import warnings
import yaml
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec # 導入 GridSpec

warnings.filterwarnings("ignore", category=UserWarning)

from dataset.datasets_wicaf_wDA_wmask import VIGORDataset
from model.models_wicaf_wlum_adapted2d_wDA_wmask_wmixstyle import CVM_VIGOR as CVM

# --- 解析度計算參數 ---
RESIZE = 512
ORIGINAL_IMG_SIZE = 640.0

# 方法一: 原始的固定解析度計算
ORIGINAL_METERS_PER_PIXEL = 72.96 / RESIZE

# 方法二: SliceMatch 論文中根據不同城市提供的解析度
CITY_ORIGINAL_RESOLUTIONS = {
    'Chicago': 0.106,
    'NewYork': 0.113,
    'SanFrancisco': 0.095,
    'Seattle': 0.101
}

def un_normalize_image(tensor_img):
    """將標準化後的 Tensor 影像還原為 PIL Image"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = tensor_img.cpu().numpy().transpose((1, 2, 0))
    np_img = std * np_img + mean
    np_img = np.clip(np_img, 0, 1)
    return Image.fromarray((np_img * 255).astype(np.uint8))

def visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 建立輸出資料夾
    if args.show_mask:
        vis_dir = os.path.join(os.path.dirname(args.model_path), 'visualizations_loc_pos_showmask')
    else:
        vis_dir = os.path.join(os.path.dirname(args.model_path), 'visualizations_loc_pos')

    os.makedirs(vis_dir, exist_ok=True)

    if not os.path.exists(args.model_path):
        print(f"錯誤: 找不到模型檔案 '{args.model_path}'")
        return

    # --- 載入模型與訓練參數 ---
    print("正在載入模型...")
    checkpoint = torch.load(args.model_path, map_location=device)

    model_args = None
    args_yaml_path = os.path.join(os.path.dirname(args.model_path), 'args.yaml')
    if os.path.exists(args_yaml_path):
        with open(args_yaml_path, 'r') as f: model_args = yaml.safe_load(f)
        print(f"成功從 {args_yaml_path} 載入模型參數。")
    elif checkpoint.get('args') is not None:
        model_args = vars(checkpoint.get('args'))
        print("成功從 checkpoint 檔案中載入模型參數。")
    else:
        print("警告: 找不到模型參數，將使用預設參數。")
        model_args = {
            'd_model': 320, 'n_layer': 2, 'vert_anchors': 16,
            'horz_anchors': 16, 'num_heads': 8, 'block_exp': 4, 'FoV': 360,
            'dropout_rate': 0.3
        }

    model = CVM(c_in_grd=3, c_in_sat=3,
                d_model=model_args.get('d_model', 320),
                n_layer=model_args.get('n_layer', 2),
                vert_anchors=model_args.get('vert_anchors', 16),
                horz_anchors=model_args.get('horz_anchors', 16),
                h=model_args.get('num_heads', 8),
                block_exp=model_args.get('block_exp', 4),
                FoV=model_args.get('FoV', 360),
                dropout_rate=model_args.get('dropout_rate', 0.3)
               ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("模型載入完成。")

    # --- 載入資料集 ---
    print("正在載入資料集...")
    val_dataset = VIGORDataset(root=args.dataset_root, split='samearea', train=False, transform=None, use_mask=args.use_mask)
    if not 0 <= args.idx < len(val_dataset):
        print(f"錯誤: 索引 {args.idx} 超出範圍。資料集大小為 {len(val_dataset)}。")
        return

    grd_tensor, sat_tensor, gt_heatmap_tensor, mask_tensor = val_dataset[args.idx]

    # --- 模型推論 ---
    grd_tensor_batch = grd_tensor.unsqueeze(0).to(device)
    sat_tensor_batch = sat_tensor.unsqueeze(0).to(device)
    mask_tensor_batch = mask_tensor.unsqueeze(0).to(device)

    print(f"正在對索引 {args.idx} 的圖像進行推論 (use_mask={args.use_mask})...")
    with torch.no_grad():
        _, pred_heatmap_tensor = model(grd_tensor_batch, sat_tensor_batch, mask_tensor_batch)

    # --- 計算雙重誤差 ---
    pred_heatmap_original = pred_heatmap_tensor.squeeze().cpu().numpy()
    gt_heatmap = gt_heatmap_tensor.squeeze().cpu().numpy()
    loc_gt = np.unravel_index(np.argmax(gt_heatmap), gt_heatmap.shape)
    loc_pred = np.unravel_index(np.argmax(pred_heatmap_original), pred_heatmap_original.shape)
    pixel_error = np.sqrt((loc_gt[0] - loc_pred[0])**2 + (loc_gt[1] - loc_pred[1])**2)

    meter_error_original = pixel_error * ORIGINAL_METERS_PER_PIXEL
    positive_sat_index = val_dataset.label[args.idx][0]
    sat_path = val_dataset.sat_list[positive_sat_index]
    try:
        city_name = sat_path.split(os.sep)[-3]
        city_res = CITY_ORIGINAL_RESOLUTIONS.get(city_name, ORIGINAL_METERS_PER_PIXEL)
        meters_per_pixel_city = city_res * (ORIGINAL_IMG_SIZE / RESIZE)

    except IndexError:
        city_name = 'unknown'
        meters_per_pixel_city = ORIGINAL_METERS_PER_PIXEL
    meter_error_city = pixel_error * meters_per_pixel_city

    # --- 產生 TXT 報告 ---
    print("推論完成，正在產生報告...")
    print(f"  - 像素誤差: {pixel_error:.2f} pixels")
    print(f"  - 公尺誤差 (原始計算): {meter_error_original:.2f} m")
    print(f"  - 公尺誤差 (城市對應, '{city_name}'): {meter_error_city:.2f} m")

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sat_filename = os.path.basename(sat_path)
    lat_lon_match = re.search(r'satellite_([0-9.-]+)_([0-9.-]+)\.png', sat_filename)
    center_lat_lon = f"({float(lat_lon_match.group(1)):.6f}, {float(lat_lon_match.group(2)):.6f})" if lat_lon_match else "無法解析"

    report_content = f"""Timestamp: {timestamp}
Model Path: {args.model_path}
Image Index: {args.idx}
Use Semantic Mask: {args.use_mask}

Satellite Filename: {sat_filename}
Detected City: {city_name}
Satellite Image Center Lat/Lon: {center_lat_lon}

GT Location (H, W): {loc_gt}
Predicted Location (H, W): {loc_pred}

Prediction error (pixels): {pixel_error:.2f} pixels
Prediction error (meters, Original Calc): {meter_error_original:.2f} m
Prediction error (meters, City-Specific Calc): {meter_error_city:.2f} m
"""
    report_filename = os.path.join(vis_dir, f'report_compared_{args.idx}_{sat_filename}.txt')
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"詳細報告已儲存至: {report_filename}")

    # --- 準備繪圖所需影像 ---
    grd_img_pil = un_normalize_image(grd_tensor)
    sat_img_pil = un_normalize_image(sat_tensor)
    input_mask_np = mask_tensor.squeeze().cpu().numpy()

    # --- 準備預測熱圖 (Prediction Heatmap) ---
    pred_heatmap = pred_heatmap_original
    min_val, max_val = pred_heatmap.min(), pred_heatmap.max()
    heatmap_normalized = (pred_heatmap - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(pred_heatmap)
    heatmap_rgba = cm.jet(heatmap_normalized)
    heatmap_rgba[:, :, 3] = 0.5

    # --- 繪製並儲存主視覺化圖 ---
    # title_text = (f'Pixel Error: {pixel_error:.2f} pixels | Meter Error: {meter_error_city:.2f} m')

    if args.show_mask:
        # --- 4 張圖的版面 (上面1張長圖，下面3張方圖) ---
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 2]) # 2行3列，第2行高度是第1行的2倍
        # fig.suptitle(title_text, fontsize=20)
        
        # 定義各個子圖的位置
        ax0 = fig.add_subplot(gs[0, :])  # 第1行，佔用所有3列 (全景圖)
        ax1 = fig.add_subplot(gs[1, 0])  # 第2行，第1列 (衛星圖)
        ax2 = fig.add_subplot(gs[1, 1])  # 第2行，第2列 (遮罩疊圖)
        ax3 = fig.add_subplot(gs[1, 2])  # 第2行，第3列 (熱力圖)
        
        # 繪製 ax0: 全景圖
        ax0.imshow(grd_img_pil)
        ax0.set_title('Ground-level Panorama', fontsize=20)
        ax0.axis('off')
        
        # 繪製 ax1: 原始衛星圖
        ax1.imshow(sat_img_pil)
        ax1.set_title('Original Satellite Image', fontsize=20)
        ax1.axis('off')

        # 準備遮罩疊圖
        sat_img_rgba = sat_img_pil.convert('RGBA')
        overlay_layer = Image.new('RGBA', sat_img_rgba.size, (0, 0, 0, 0))
        mask_color = (255, 0, 0, 100) # 半透明紅色
        mask_pil_for_paste = Image.fromarray((input_mask_np * 255).astype(np.uint8), mode='L')
        overlay_layer.paste(mask_color, mask=mask_pil_for_paste)
        mask_overlay_image = Image.alpha_composite(sat_img_rgba, overlay_layer)
        
        # 繪製 ax2: 遮罩疊圖
        ax2.imshow(mask_overlay_image)
        ax2.set_title('Input Mask on Satellite', fontsize=20)
        ax2.axis('off')
        
        # 繪製 ax3: 熱力圖
        ax3.imshow(sat_img_pil)
        ax3.imshow(heatmap_rgba)
        ax3.scatter(loc_gt[1], loc_gt[0], color='lime', marker='o', edgecolor='black', s=250, linewidth=2.5, label='Ground Truth')
        ax3.scatter(loc_pred[1], loc_pred[0], color='gold', marker='^', edgecolor='black', s=250, linewidth=2.5, label='Prediction')
        ax3.set_title('Prediction Heatmap on Satellite', fontsize=20)
        ax3.legend(loc='best')
        ax3.axis('off')

    else:
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        # fig.suptitle(title_text, fontsize=20)

        # 繪製3張子圖
        axes[0].imshow(grd_img_pil); axes[0].set_title('Ground-level Panorama', fontsize=20); axes[0].axis('off')
        axes[1].imshow(sat_img_pil); axes[1].set_title('Original Satellite Image', fontsize=20); axes[1].axis('off')

        axes[2].imshow(sat_img_pil)
        axes[2].imshow(heatmap_rgba)
        axes[2].scatter(loc_gt[1], loc_gt[0], color='lime', marker='o', edgecolor='black', s=250, linewidth=2.5, label='Ground Truth')
        axes[2].scatter(loc_pred[1], loc_pred[0], color='gold', marker='^', edgecolor='black', s=250, linewidth=2.5, label='Prediction')
        axes[2].set_title('Prediction Heatmap on Satellite', fontsize=20); axes[2].legend(loc='best'); axes[2].axis('off')

        # 繪製顏色條 (Colorbar)
        norm_for_colorbar = plt.Normalize(vmin=pred_heatmap_original.min(), vmax=pred_heatmap_original.max())
        mappable = cm.ScalarMappable(norm=norm_for_colorbar, cmap='jet')
        fig.colorbar(mappable, ax=axes[2], fraction=0.046, pad=0.04, label='Heatmap Score')
    

    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    
    vis_filename = os.path.join(vis_dir, f'visualization_compared_{args.idx}_{sat_filename}')
    plt.savefig(vis_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"主要視覺化圖像已儲存至: {vis_filename}")

    print("\n處理完畢！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="為 VIGOR 資料集產生模型預測的視覺化熱圖，並可選擇性地輸出疊圖。")

    
    parser.add_argument('--model_path', type=str, required=False, 
                        default='runs/samearea/pos/27.75/best.pt',
                        help='已訓練模型的路徑 (例如: runs/VIGOR/.../best.pt)。')
    
    parser.add_argument('--dataset_root', type=str, required=False, default='/home/lhl/Documents/VIGOR'
, 
                        help='VIGOR 資料集的根目錄路徑。')

    parser.add_argument('--idx', type=int, required=False, default=30,
                        help='要進行視覺化的驗證集圖像索引。')

    parser.add_argument('--output_dir', type=str, default='./visualization_output',
                        help='儲存輸出圖像和報告的資料夾。')


    parser.add_argument('--use_mask', type=lambda x: (str(x).lower() == 'true'), default=True, help='視覺化時是否載入並將 semantic mask 傳遞給模型 (True/False)，建議保持為 True。')

    parser.add_argument('--show_mask', action='store_true', help='若啟用此參數，將在主圖中顯示衛星圖與遮罩的疊圖 (共4張子圖)，否則只顯示3張。')

    args = parser.parse_args()
    visualize(args)