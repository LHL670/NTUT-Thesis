

import os
from torch.utils.data import Dataset
import PIL.Image
import torch
import numpy as np
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms.functional as TF
import math
import torchvision.transforms as transforms
import cv2

torch.manual_seed(17)
np.random.seed(0)
random.seed(17)

# ---------------------------------------------------------------------------------
# VIGOR

class VIGORDataset(Dataset):
    def __init__(self, root, label_root='splits__corrected', split='samearea', train=True, transform=None, pos_only=True, ori_noise=180, random_orientation=None, use_mask=True):
        self.root = root
        self.label_root = label_root
        self.split = split
        self.train = train
        self.pos_only = pos_only 
        self.ori_noise = ori_noise
        self.random_orientation = random_orientation
        self.use_mask = use_mask

        if self.train:
            self.grdimage_transform = transforms.Compose([
                transforms.Resize([320, 640]),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
            ])
            self.satimage_transform = transforms.Compose([
                transforms.Resize([512, 512]),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
            ])
        else:
            self.grdimage_transform = transforms.Compose([
                transforms.Resize([320, 640]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.satimage_transform = transforms.Compose([
                transforms.Resize([512, 512]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        if self.split == 'samearea':
            self.city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        elif self.split == 'crossarea':
            self.city_list = ['NewYork', 'Seattle'] if self.train else ['SanFrancisco', 'Chicago']

        self.sat_list = []
        self.sat_index_dict = {}
        idx = 0
        for city in self.city_list:
            with open(os.path.join(self.root, label_root, city, 'satellite_list.txt'), 'r') as file:
                for line in file:
                    self.sat_list.append(os.path.join(self.root, city, 'satellite', line.strip()))
                    self.sat_index_dict[line.strip()] = idx
                    idx += 1
        self.sat_list = np.array(self.sat_list)
        self.sat_data_size = len(self.sat_list)

        self.grd_list = []
        self.label = []
        self.delta = []
        for city in self.city_list:
            label_fname = os.path.join(self.root, self.label_root, city,
                'same_area_balanced_train__corrected.txt' if self.train else 'same_area_balanced_test__corrected.txt')
            with open(label_fname, 'r') as file:
                for line in file:
                    data = np.array(line.split(' '))
                    self.label.append([self.sat_index_dict[data[i]] for i in [1, 4, 7, 10]])
                    self.delta.append(np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float))
                    self.grd_list.append(os.path.join(self.root, city, 'panorama', data[0]))
        self.data_size = len(self.grd_list)
        self.label = np.array(self.label)
        self.delta = np.array(self.delta)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        # 讀取街景影像
        try:
            grd = PIL.Image.open(self.grd_list[idx]).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load ground image {self.grd_list[idx]}. Error: {e}. Returning a blank image.")
            grd = PIL.Image.new('RGB', (640, 320))

        if self.train and not self.pos_only:
            if random.random() < 0.5:
                pos_index = 0  
            else:
                pos_index = random.randint(1, 3) 
            
            col_offset_check = self.delta[idx, pos_index][1]
            row_offset_check = self.delta[idx, pos_index][0]
            if np.abs(col_offset_check) >= 320 or np.abs(row_offset_check) >= 320:
                pos_index = 0 
        else:
            pos_index = 0
            
        sat_path = self.sat_list[self.label[idx][pos_index]]
        row_offset, col_offset = self.delta[idx, pos_index]


        # 讀取選定的衛星影像
        try:
            sat = PIL.Image.open(sat_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load satellite image {sat_path}. Error: {e}. Returning a blank image.")
            sat = PIL.Image.new('RGB', (512, 512))
            

        # 根據 use_mask 和路徑規則，載入並處理語意遮罩
        if self.use_mask:
            sat_filename = os.path.basename(sat_path)
            parts = sat_path.split(os.sep)
            city_name = parts[-3] if len(parts) >= 3 else None
            if city_name:
                mask_filename = sat_filename.replace('.png', '_mask.tif')
                mask_path = os.path.join(self.root, city_name, 'point_prompt_mask', mask_filename)
                
                try:
                    mask_np = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    if mask_np is None: raise FileNotFoundError("cv2.imread returned None")
                    binary_mask_np = (mask_np > 0).astype(np.float32)
                    mask_pil = PIL.Image.fromarray(binary_mask_np)
                except Exception as e:
                    print(f"Warning: Could not load or process mask {mask_path}. Error: {e}. Returning a blank mask.")
                    mask_pil = PIL.Image.new('L', sat.size, 0)
            else:
                print(f"Warning: Could not determine city from path {sat_path}. Returning a blank mask.")
                mask_pil = PIL.Image.new('L', sat.size, 0)
        else:
            mask_pil = PIL.Image.new('L', sat.size, 0)

        # 生成 GT 熱圖 (Ground Truth Heatmap)
        width_raw, height_raw = sat.size
        sat_resized_shape = self.satimage_transform.transforms[0].size
        height, width = sat_resized_shape[0], sat_resized_shape[1]
        
        row_offset_resized = np.round(row_offset / height_raw * height)
        col_offset_resized = np.round(col_offset / width_raw * width)

        gt = np.zeros([1, height, width], dtype=np.float32)
        x, y = np.meshgrid(np.linspace(-width/2 + col_offset_resized, width/2 + col_offset_resized, width),
                           np.linspace(-height/2 - row_offset_resized, height/2 - row_offset_resized, height))
        d = np.sqrt(x * x + y * y)
        sigma = 4
        gt[0] = np.exp(-((d)**2) / (2 * sigma**2))
        
        gt_tensor = torch.from_numpy(gt)

        # 對衛星圖、GT熱圖 和 MASK 進行同步的幾何增強
        if self.train:
            if random.random() > 0.5:
                sat = TF.hflip(sat)
                gt_tensor = TF.hflip(gt_tensor)
                mask_pil = TF.hflip(mask_pil) 

            if random.random() > 0.5:
                sat = TF.vflip(sat)
                gt_tensor = TF.vflip(gt_tensor)
                mask_pil = TF.vflip(mask_pil)

            angle = random.uniform(-10, 10)
            translate = (random.randint(-15, 15), random.randint(-15, 15))
            scale = random.uniform(0.9, 1.1)
            shear = (random.uniform(-5, 5), random.uniform(-5, 5))

            sat = TF.affine(sat, angle, translate, scale, shear, interpolation=TF.InterpolationMode.BILINEAR)
            gt_tensor = TF.affine(gt_tensor, angle, translate, scale, shear, interpolation=TF.InterpolationMode.BILINEAR)
            mask_pil = TF.affine(mask_pil, angle, translate, scale, shear, interpolation=TF.InterpolationMode.NEAREST)

        grd = self.grdimage_transform(grd)
        sat = self.satimage_transform(sat)

        mask_transform = transforms.Compose([
            transforms.Resize(self.satimage_transform.transforms[0].size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        mask_tensor = mask_transform(mask_pil)

        # 街景環景圖的隨機旋轉
        if self.train and self.ori_noise > 0:
            if self.ori_noise >= 180:
                rotation = np.random.uniform(0.0, 1.0)
            else:
                rotation_range = self.ori_noise / 360
                rotation = np.random.uniform(-rotation_range, rotation_range)
            grd = torch.roll(grd, int(torch.round(torch.tensor(rotation) * grd.size(2)).item()), dims=2)

        return grd, sat, gt_tensor, mask_tensor