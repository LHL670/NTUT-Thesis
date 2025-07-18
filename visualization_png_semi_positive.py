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
import torchvision.transforms as transforms
import cv2
from dataset.datasets_wicaf_wDA_wmask import VIGORDataset
from model.models_wicaf_wlum_adapted2d_wDA_wmask_wmixstyle import CVM_VIGOR as CVM
warnings.filterwarnings("ignore", category=UserWarning)

# --- 解析度計算參數 ---
RESIZE = 512
ORIGINAL_IMG_SIZE = 640.0
ORIGINAL_METERS_PER_PIXEL = 72.96 / RESIZE
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

def generate_gt_heatmap(sat_pil_size, delta_offset, sat_resized_shape):
    """根據偏移量生成 Ground Truth 熱圖"""
    width_raw, height_raw = sat_pil_size
    height, width = sat_resized_shape[0], sat_resized_shape[1]
    
    row_offset, col_offset = delta_offset
    row_offset_resized = np.round(row_offset / height_raw * height)
    col_offset_resized = np.round(col_offset / width_raw * width)

    gt = np.zeros([1, height, width], dtype=np.float32)
    x, y = np.meshgrid(np.linspace(-width/2 + col_offset_resized, width/2 + col_offset_resized, width),
                       np.linspace(-height/2 - row_offset_resized, height/2 - row_offset_resized, height))
    d = np.sqrt(x * x + y * y)
    sigma = 4
    gt[0] = np.exp(-((d)**2) / (2 * sigma**2))
    
    return torch.from_numpy(gt)

def visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 建立輸出資料夾
    vis_dir = os.path.join(os.path.dirname(args.model_path), 'visualization_semi_positive')
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

    # --- 載入資料集以獲取資料路徑和標籤 ---
    print("正在載入資料集元數據...")
    val_dataset = VIGORDataset(root=args.dataset_root, split='samearea', train=False, transform=None, use_mask=args.use_mask)
    if not 0 <= args.idx < len(val_dataset):
        print(f"錯誤: 索引 {args.idx} 超出範圍。資料集大小為 {len(val_dataset)}。")
        return

    # --- 定義驗證模式下的 Transform ---
    grd_transform = transforms.Compose([
        transforms.Resize([320, 640]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    sat_transform = transforms.Compose([
        transforms.Resize([512, 512]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize([512, 512], interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    # --- 獲取街景圖 (只需一次) ---
    grd_path = val_dataset.grd_list[args.idx]
    grd_pil = Image.open(grd_path).convert('RGB')
    grd_tensor = grd_transform(grd_pil)
    grd_tensor_batch = grd_tensor.unsqueeze(0).to(device)
    
    # --- 獲取 4 組衛星圖的資訊 ---
    sat_indices = val_dataset.label[args.idx]
    delta_offsets = val_dataset.delta[args.idx]
    
    results = []
    print(f"正在對索引 {args.idx} 的 1 張街景圖與 4 張衛星圖進行推論...")

    # --- 迴圈處理 4 組衛星圖 ---
    for i in range(4):
        print(f"  處理第 {i+1}/4 組衛星圖...")
        # 手動載入並處理每張衛星圖
        sat_index = sat_indices[i]
        delta = delta_offsets[i]
        sat_path = val_dataset.sat_list[sat_index]

        sat_pil = Image.open(sat_path).convert('RGB')
        
        # 處理 Mask
        if args.use_mask:
            sat_filename = os.path.basename(sat_path)
            parts = sat_path.split(os.sep)
            city_name_from_path = parts[-3]
            mask_filename = sat_filename.replace('.png', '_mask.tif')
            mask_path = os.path.join(val_dataset.root, city_name_from_path, 'point_prompt_mask', mask_filename)
            try:
                mask_np = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask_np is None: raise FileNotFoundError("cv2.imread returned None")
                binary_mask_np = (mask_np > 0).astype(np.float32)
                mask_pil = Image.fromarray(binary_mask_np)
            except Exception as e:
                mask_pil = Image.new('L', sat_pil.size, 0)
        else:
            mask_pil = Image.new('L', sat_pil.size, 0)

        # 產生 GT Heatmap
        gt_heatmap_tensor = generate_gt_heatmap(sat_pil.size, delta, sat_transform.transforms[0].size)

        # 轉換為 Tensor
        sat_tensor = sat_transform(sat_pil)
        mask_tensor = mask_transform(mask_pil)

        # 準備批次進行推論
        sat_tensor_batch = sat_tensor.unsqueeze(0).to(device)
        mask_tensor_batch = mask_tensor.unsqueeze(0).to(device)

        # 模型推論
        with torch.no_grad():
            _, pred_heatmap_tensor = model(grd_tensor_batch, sat_tensor_batch, mask_tensor_batch)

        # 計算誤差
        pred_heatmap_numpy = pred_heatmap_tensor.squeeze().cpu().numpy()
        gt_heatmap_numpy = gt_heatmap_tensor.squeeze().numpy()
        loc_gt = np.unravel_index(np.argmax(gt_heatmap_numpy), gt_heatmap_numpy.shape)
        loc_pred = np.unravel_index(np.argmax(pred_heatmap_numpy), pred_heatmap_numpy.shape)
        pixel_error = np.sqrt((loc_gt[0] - loc_pred[0])**2 + (loc_gt[1] - loc_pred[1])**2)

        try:
            city_name = sat_path.split(os.sep)[-3]
            city_res = CITY_ORIGINAL_RESOLUTIONS.get(city_name, ORIGINAL_METERS_PER_PIXEL)
            meters_per_pixel_city = city_res * (ORIGINAL_IMG_SIZE / RESIZE)
        except IndexError:
            city_name = 'unknown'
            meters_per_pixel_city = ORIGINAL_METERS_PER_PIXEL
        meter_error_city = pixel_error * meters_per_pixel_city

        # 儲存該次結果
        results.append({
            'sat_path': sat_path,
            'sat_pil': un_normalize_image(sat_tensor),
            'pred_heatmap': pred_heatmap_numpy,
            'loc_gt': loc_gt,
            'loc_pred': loc_pred,
            'pixel_error': pixel_error,
            'meter_error': meter_error_city,
            'city_name': city_name,
            'type': 'Positive' if i == 0 else f'Semi-Positive {i}'
        })

    # --- 產生 TXT 報告 ---
    print("推論完成，正在產生報告...")
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report_content = f"""Timestamp: {timestamp}
Model Path: {args.model_path}
Ground Image Index: {args.idx}
Ground Image Path: {grd_path}
Use Semantic Mask: {args.use_mask}
"""
    
    for i, res in enumerate(results):
        sat_filename = os.path.basename(res['sat_path'])
        lat_lon_match = re.search(r'satellite_([0-9.-]+)_([0-9.-]+)\.png', sat_filename)
        center_lat_lon = f"({float(lat_lon_match.group(1)):.6f}, {float(lat_lon_match.group(2)):.6f})" if lat_lon_match else "無法解析"

        report_content += f"""
-----------------------------------------
Prediction for Satellite {i+1}/4 ({res['type']})
-----------------------------------------
Satellite Filename: {sat_filename}
Detected City: {res['city_name']}
Satellite Image Center Lat/Lon: {center_lat_lon}

GT Location (H, W): {res['loc_gt']}
Predicted Location (H, W): {res['loc_pred']}

Prediction error (pixels): {res['pixel_error']:.2f} pixels
Prediction error (meters, City-Specific Calc): {res['meter_error']:.2f} m
"""

    report_filename = os.path.join(vis_dir, f'report_{args.idx}_{os.path.basename(grd_path)}.txt')
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"詳細報告已儲存至: {report_filename}")


    fig = plt.figure(figsize=(18, 24))
    # fig.suptitle(f'Visual Analysis for G-IDX: {args.idx} (1 Ground vs 4 Satellite Predictions)', fontsize=20, y=0.98)

    ax_pano = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax_pano.imshow(grd_pil)
    ax_pano.set_title('Ground-level Panorama', fontsize=20)
    ax_pano.axis('off')

    axes_sat = [
        plt.subplot2grid((3, 2), (1, 0)),
        plt.subplot2grid((3, 2), (1, 1)),
        plt.subplot2grid((3, 2), (2, 0)),
        plt.subplot2grid((3, 2), (2, 1))
    ]

    for i, ax in enumerate(axes_sat):
        res = results[i]
        
        # --- 準備預測熱圖 (Prediction Heatmap) ---
        pred_heatmap = res['pred_heatmap']

        min_val, max_val = pred_heatmap.min(), pred_heatmap.max()
        heatmap_normalized = (pred_heatmap - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(pred_heatmap)
        
        heatmap_rgba = cm.jet(heatmap_normalized)
        
        heatmap_rgba[:, :, 3] = 0.5 
        ax.imshow(res['sat_pil'])
        ax.imshow(heatmap_rgba)
        ax.scatter(res['loc_gt'][1], res['loc_gt'][0], color='lime', marker='o', edgecolor='black', s=250, linewidth=2.5, label='Ground Truth')
        ax.scatter(res['loc_pred'][1], res['loc_pred'][0], color='gold', marker='^', edgecolor='black', s=250, linewidth=2.5, label='Prediction')
        
        title = f"{i+1}. {res['type']}\nPixel Error: {res['pixel_error']:.2f} pixels | Meter Error: {res['meter_error']:.2f}m"
        ax.set_title(title, fontsize=20)
        ax.legend(loc='best')
        ax.axis('off')

    plt.subplots_adjust(
        top=0.96,      
        bottom=0.01,   
        left=0.10,     
        right=0.90,    
        hspace=0.1,   
        wspace=0.1    
    )
    vis_filename = os.path.join(vis_dir, f'visualization_{args.idx}_{os.path.basename(grd_path)}.png')
    plt.savefig(vis_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"主要視覺化圖像已儲存至: {vis_filename}")
    print("\n處理完畢！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="為 VIGOR 資料集產生 1 對 4 的視覺化結果 (1張街景對應4張衛星圖預測)。")

    parser.add_argument('--model_path', type=str, required=False, 
                        default='runs/samearea/semi/45.48_semi_wicaf_wlum_adapted2d_wDA20250703_151501_samearea_nL2_T0.1_wd0.001_useMaskTrue_useMixStyleTrue_mix_random/best.pt',
                        help='已訓練模型的路徑 (例如: runs/VIGOR/.../best.pt)。')
    
    parser.add_argument('--dataset_root', type=str, required=False, default='/home/lhl/Documents/VIGOR', 
                        help='VIGOR 資料集的根目錄路徑。')

    parser.add_argument('--idx', type=int, required=False, default=30,
                        help='要進行視覺化的驗證集圖像索引 (對應到街景圖)。')


    parser.add_argument('--use_mask', type=lambda x: (str(x).lower() == 'true'), default=True, 
                        help='視覺化時是否載入並將 semantic mask 傳遞給模型 (True/False)。')

    args = parser.parse_args()
    visualize(args)