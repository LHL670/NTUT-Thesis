
#  1. 產生總覽圖 (原圖, Grad-CAM+定位點, 最終層注意力圖)
#       2. 產生逐層注意力分析圖 (Transformer 每一層的 Attention Map) -

import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from tqdm import tqdm
from dataset.datasets_wicaf_wDA_wmask import VIGORDataset
from model.models_wicaf_wlum_adapted2d_wDA_wmask_wmixstyle_gradcam import CVM_VIGOR as CVM

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """將標準化的 Tensor 還原為可用於顯示的 Numpy 影像"""
    tensor = tensor.cpu()
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.squeeze(0).permute(1, 2, 0).numpy()

def visualize_final_attention(attention_maps_list, target_image_unnorm, vert_anchors, horz_anchors):
    """(原有的) 處理並視覺化 Transformer 最終層的注意力圖"""
    if not attention_maps_list:
        return target_image_unnorm

    attention_map = attention_maps_list[-1].squeeze(0).cpu().detach()
    attention_map = torch.mean(attention_map, dim=0)
    attention_map = torch.mean(attention_map, dim=0)
    
    num_tokens = vert_anchors * horz_anchors
    if attention_map.shape[0] != num_tokens:
        print(f"Token 數量不匹配! 期望: {num_tokens}, 實際: {attention_map.shape[0]}")
        return target_image_unnorm
        
    attention_map_2d = attention_map.view(vert_anchors, horz_anchors)
    attention_map_2d = (attention_map_2d - attention_map_2d.min()) / (attention_map_2d.max() - attention_map_2d.min())
    attention_map_2d = attention_map_2d.numpy()
    
    target_h, target_w, _ = target_image_unnorm.shape
    attention_heatmap = cv2.resize(attention_map_2d, (target_w, target_h))
    
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * attention_heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    superimposed_img = (heatmap_colored * 0.4 + target_image_unnorm * 255 * 0.6).astype(np.uint8)
    return superimposed_img

def visualize_attention_by_layer(att_maps_g2s, att_maps_s2g, grd_img_unnorm, sat_img_unnorm, model_args, output_filename):
    """處理並視覺化 Transformer 每一層的注意力圖"""
    n_layers = model_args.n_layer
    vert_anchors, horz_anchors = model_args.vert_anchors, model_args.horz_anchors
    
    fig, axs = plt.subplots(2, n_layers + 1, figsize=(5 * (n_layers + 1), 10))
    
    # 顯示原始影像
    axs[0, 0].imshow(grd_img_unnorm)
    axs[0, 0].set_title('Original Ground View')
    axs[0, 0].axis('off')
    
    axs[1, 0].imshow(sat_img_unnorm)
    axs[1, 0].set_title('Original Satellite View')
    axs[1, 0].axis('off')

    # 逐層顯示注意力圖
    for i in range(n_layers):
        # --- Sat-to-Ground Attention (顯示在 Ground-View 上) ---
        att_map_s2g_i = att_maps_s2g[i].squeeze(0).cpu().detach()
        att_map_s2g_i = torch.mean(att_map_s2g_i, dim=0) # 平均 head
        
        # 找到被關注度最高的 token
        max_attn_token_idx_s2g = torch.argmax(torch.sum(att_map_s2g_i, dim=0))
        att_map_s2g_i = att_map_s2g_i[max_attn_token_idx_s2g, :]
        
        att_map_2d_s2g = att_map_s2g_i.view(vert_anchors, horz_anchors)
        att_map_2d_s2g = (att_map_2d_s2g - att_map_2d_s2g.min()) / (att_map_2d_s2g.max() - att_map_2d_s2g.min())
        
        h, w, _ = grd_img_unnorm.shape
        heatmap_s2g = cv2.resize(att_map_2d_s2g.numpy(), (w, h))
        heatmap_s2g_color = cv2.applyColorMap(np.uint8(255 * heatmap_s2g), cv2.COLORMAP_JET)
        heatmap_s2g_color = cv2.cvtColor(heatmap_s2g_color, cv2.COLOR_BGR2RGB)
        
        superimposed_s2g = (heatmap_s2g_color * 0.5 + grd_img_unnorm * 255 * 0.5).astype(np.uint8)
        axs[0, i + 1].imshow(superimposed_s2g)
        axs[0, i + 1].set_title(f'S2G Attention - Layer {i+1}')
        axs[0, i + 1].axis('off')

        # --- Ground-to-Satellite Attention (顯示在 Sat-View 上) ---
        att_map_g2s_i = att_maps_g2s[i].squeeze(0).cpu().detach()
        att_map_g2s_i = torch.mean(att_map_g2s_i, dim=0) # 平均 head
        
        max_attn_token_idx_g2s = torch.argmax(torch.sum(att_map_g2s_i, dim=0))
        att_map_g2s_i = att_map_g2s_i[max_attn_token_idx_g2s, :]
        
        att_map_2d_g2s = att_map_g2s_i.view(vert_anchors, horz_anchors)
        att_map_2d_g2s = (att_map_2d_g2s - att_map_2d_g2s.min()) / (att_map_2d_g2s.max() - att_map_2d_g2s.min())

        h, w, _ = sat_img_unnorm.shape
        heatmap_g2s = cv2.resize(att_map_2d_g2s.numpy(), (w, h))
        heatmap_g2s_color = cv2.applyColorMap(np.uint8(255 * heatmap_g2s), cv2.COLORMAP_JET)
        heatmap_g2s_color = cv2.cvtColor(heatmap_g2s_color, cv2.COLOR_BGR2RGB)
        
        superimposed_g2s = (heatmap_g2s_color * 0.5 + sat_img_unnorm * 255 * 0.5).astype(np.uint8)
        axs[1, i + 1].imshow(superimposed_g2s)
        axs[1, i + 1].set_title(f'G2S Attention - Layer {i+1}')
        axs[1, i + 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)

# --- Grad-CAM 的模型包裝器 ---
class G2S_ModelWrapper(torch.nn.Module):
    def __init__(self, model, sat_tensor, mask_tensor):
        super(G2S_ModelWrapper, self).__init__()
        self.model = model
        self.sat_tensor = sat_tensor
        self.mask_tensor = mask_tensor

    def forward(self, grd_tensor):
        return self.model(grd_tensor, self.sat_tensor, self.mask_tensor)[0]

class S2G_ModelWrapper(torch.nn.Module):
    def __init__(self, model, grd_tensor, mask_tensor):
        super(S2G_ModelWrapper, self).__init__()
        self.model = model
        self.grd_tensor = grd_tensor
        self.mask_tensor = mask_tensor

    def forward(self, sat_tensor):
        return self.model(self.grd_tensor, sat_tensor, self.mask_tensor)[0]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("正在載入模型...")
    checkpoint = torch.load(args.resume, map_location=device)
    
    config_path = os.path.join(os.path.dirname(args.resume), 'args.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            model_args = argparse.Namespace(**yaml.safe_load(f))
    else:
        # 如果找不到 config file，從 checkpoint 或預設值載入
        model_args = checkpoint.get('args', argparse.Namespace())
        setattr(model_args, 'd_model', getattr(model_args, 'd_model', 320))
        setattr(model_args, 'n_layer', getattr(model_args, 'n_layer', 2))
        setattr(model_args, 'vert_anchors', getattr(model_args, 'vert_anchors', 16))
        setattr(model_args, 'horz_anchors', getattr(model_args, 'horz_anchors', 16))
        setattr(model_args, 'num_heads', getattr(model_args, 'num_heads', 8))
        setattr(model_args, 'block_exp', getattr(model_args, 'block_exp', 4))
        setattr(model_args, 'FoV', getattr(model_args, 'FoV', 360))
        setattr(model_args, 'use_mixstyle', getattr(model_args, 'use_mixstyle', False))

    model = CVM(
        d_model=model_args.d_model, n_layer=model_args.n_layer,
        vert_anchors=model_args.vert_anchors, horz_anchors=model_args.horz_anchors,
        h=model_args.num_heads, block_exp=model_args.block_exp,
        FoV=model_args.FoV, use_mixstyle=model_args.use_mixstyle
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("模型載入完成。")

    val_dataset = VIGORDataset(root=args.dataset_root, split='samearea', train=False, use_mask=True)
    
    output_dir = os.path.join(os.path.dirname(args.resume), 'visualizations_cross_attention')
    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(args.img_idx, args.img_idx + args.num_samples), desc="Processing images"):

        grd, sat, gt_heatmap_tensor, mask_tensor = val_dataset[i]
        
        grd_tensor = grd.unsqueeze(0).to(device)
        sat_tensor = sat.unsqueeze(0).to(device)
        mask_tensor = mask_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits, pred_heatmap, _, att_maps_g2s, att_maps_s2g = model(
                grd_tensor, sat_tensor, mask_tensor, return_attention=True
            )
        
        gt_heatmap = gt_heatmap_tensor.squeeze().cpu().numpy()
        pred_heatmap_numpy = pred_heatmap.squeeze().cpu().detach().numpy()
        loc_gt = np.unravel_index(np.argmax(gt_heatmap), gt_heatmap.shape)
        loc_pred = np.unravel_index(np.argmax(pred_heatmap_numpy), pred_heatmap_numpy.shape)
        pixel_error = np.sqrt((loc_gt[0] - loc_pred[0])**2 + (loc_gt[1] - loc_pred[1])**2)
        
        # 準備影像與 Grad-CAM
        grd_img_unnorm = unnormalize(grd_tensor)
        sat_img_unnorm = unnormalize(sat_tensor)
        
        pred_max_idx = torch.argmax(logits, dim=1).item()
        targets = [ClassifierOutputTarget(pred_max_idx)]
        target_layer_grd = [model.grd_adapt_conv]
        target_layer_sat = [model.sat_adapt_conv]
        
        wrapped_model_g2s = G2S_ModelWrapper(model, sat_tensor, mask_tensor)
        cam_grd = GradCAM(model=wrapped_model_g2s, target_layers=target_layer_grd)
        grayscale_cam_grd = cam_grd(input_tensor=grd_tensor, targets=targets)[0, :]
        cam_image_grd = show_cam_on_image(grd_img_unnorm, grayscale_cam_grd, use_rgb=True)
        
        wrapped_model_s2g = S2G_ModelWrapper(model, grd_tensor, mask_tensor)
        cam_sat = GradCAM(model=wrapped_model_s2g, target_layers=target_layer_sat)
        grayscale_cam_sat = cam_sat(input_tensor=sat_tensor, targets=targets)[0, :]
        cam_image_sat = show_cam_on_image(sat_img_unnorm, grayscale_cam_sat, use_rgb=True)

        # 準備最終層注意力圖
        attention_on_sat = visualize_final_attention(att_maps_g2s, sat_img_unnorm, model_args.vert_anchors, model_args.horz_anchors)
        attention_on_grd = visualize_final_attention(att_maps_s2g, grd_img_unnorm, model_args.vert_anchors, model_args.horz_anchors)
        
        # 1. 取得影像的原始尺寸
        grd_h, grd_w, _ = grd_img_unnorm.shape
        sat_h, sat_w, _ = sat_img_unnorm.shape

        # 2. 計算能讓寬度對齊的理想長寬比
        grd_aspect = grd_w / (grd_h + 1e-6)
        sat_aspect = sat_w / (sat_h + 1e-6)
        fig_aspect_ratio = (grd_aspect + 2 * sat_aspect) / 2

        # 3. 設定一個基礎高度，並計算出對應的寬度
        fig_height_inches = 15  # 單位為英寸，可視情況調整
        fig_width_inches = fig_height_inches * fig_aspect_ratio * (grd_h / (sat_h + 1e-6))

        fig, axs = plt.subplots(
            2, 3,
            figsize=(fig_width_inches, fig_height_inches), # 使用動態計算的尺寸
            gridspec_kw={'height_ratios': [1, 2]}  # 高度比例維持不變
            
        )
        
        # fig.suptitle(f"Full Visualization for Index: {i} | Pixel Error: {pixel_error:.2f}", fontsize=20)

        # Row 1: Ground-centric
        axs[0, 0].imshow(grd_img_unnorm)
        axs[0, 0].set_title('Original Ground View', fontsize=14)
        axs[0, 0].axis('off')
        
        axs[0, 1].imshow(cam_image_grd)
        axs[0, 1].set_title('Ground View Grad-CAM', fontsize=14)
        axs[0, 1].axis('off')

        axs[0, 2].imshow(attention_on_grd)
        axs[0, 2].set_title('Sat-to-Ground Attention', fontsize=14)
        axs[0, 2].axis('off')

        # Row 2: Satellite-centric
        axs[1, 0].imshow(sat_img_unnorm)
        axs[1, 0].set_title('Original Satellite View', fontsize=14)
        axs[1, 0].axis('off')

        axs[1, 1].imshow(cam_image_sat)
        axs[1, 1].set_title('Satellite View Grad-CAM with Localization', fontsize=14)
        axs[1, 1].axis('off')
        
        axs[1, 2].imshow(attention_on_sat)
        axs[1, 2].set_title('Ground-to-Sat Attention', fontsize=14)
        axs[1, 2].axis('off')
        
        # 在 Grad-CAM 圖上疊加定位點
        axs[1, 1].scatter(loc_gt[1], loc_gt[0], color='lime', marker='o', edgecolor='black', s=250, linewidth=2.5, label='Ground Truth')
        axs[1, 1].scatter(loc_pred[1], loc_pred[0], color='gold', marker='^', edgecolor='black', s=250, linewidth=2.5, label='Prediction')
        axs[1, 1].legend(loc='best')
        
        plt.tight_layout(rect=[0, 0.15, 1, 0.97])
        
        output_filename = os.path.join(output_dir, f'cross_visualization_{i}.png')
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"完整視覺化結果已儲存至: {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate ultimate visualizations for model analysis.")
    parser.add_argument('--resume', type=str, default='runs/samearea/pos/27.75/best.pt', help='Path to the trained model checkpoint (.pt file)')
    parser.add_argument('--dataset_root', type=str, default='/home/lhl/Documents/VIGOR', help='Root directory of VIGOR dataset')
    parser.add_argument('--img_idx', type=int, default=5606, help='Start index of the image in the validation dataset to visualize')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to visualize')
    args = parser.parse_args()
    
    main(args)