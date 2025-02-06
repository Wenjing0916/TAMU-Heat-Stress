import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from torchvision import transforms
import os
import tifffile
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms.functional as TF
import random
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, spatial_data_dir, temporal_data_csv, output_data_dir, T_in, T_out):
        self.spatial_data = []
        self.output_data = []
        self.spatial_data_max = None
        self.spatial_data_min = None
        spatial_files = sorted(os.listdir(spatial_data_dir))[:4]
        for spatial_file in spatial_files:
            spatial_image = np.array(Image.open(os.path.join(spatial_data_dir, spatial_file))).astype(np.float32)
            spatial_image = torch.tensor(spatial_image).unsqueeze(0)  
            spatial_image, self.spatial_data_max, self.spatial_data_min = self.maxminscaler_3d(spatial_image)  
            spatial_image = TF.crop(spatial_image, top=0, left=0, height=3982, width=3739)
            self.spatial_data.append(spatial_image)
        self.spatial_data = torch.cat(self.spatial_data, dim=0)

        temporal_data_df = pd.read_csv(temporal_data_csv).iloc[:, 1:]  
        self.temporal_data = temporal_data_df.select_dtypes(include=[np.number]).fillna(0).astype(np.float32).values

        self.temporal_data_max = np.max(self.temporal_data, axis=0)  
        self.temporal_data_min = np.min(self.temporal_data, axis=0)  

        num_samples_possible = (self.temporal_data.shape[0] // 336)
        self.temporal_data = self.temporal_data[:num_samples_possible * 336].reshape(-1, 336, 7)  
        self.temporal_data = self.normalize_temporal_data(self.temporal_data)

        self.output_data_paths = [os.path.join(output_data_dir, f) for f in sorted(os.listdir(output_data_dir))]
        for output_path in self.output_data_paths:
            output_image = np.array(Image.open(output_path)).astype(np.float32)
            self.output_data.append(output_image)
        self.T_in = T_in
        self.T_out = T_out
        self.utci_max = None  
        self.utci_min = None  

        print("CustomDataset initialized successfully.")
        print(f"Number of spatial images: {len(spatial_files)}")
        print(f"Temporal data shape after reshape: {self.temporal_data.shape}")
        print(f"Number of output images (UTCI): {len(self.output_data_paths)}")
        print(f"T_in: {T_in}, T_out: {T_out}")

        self.compute_utci_global_max_min()
        self.num_samples = self.temporal_data.shape[0] * (336 - (self.T_in + self.T_out - 1))
        print(f"Calculated num_samples: {self.num_samples}")

        if self.num_samples <= 0:
            raise ValueError("Not enough time steps to generate input and output sequences")

    def normalize_temporal_data(self, temporal_data):
        normalized_temporal_data = (temporal_data - self.temporal_data_min) / (self.temporal_data_max - self.temporal_data_min)
        wind_direction_index = 4
        normalized_temporal_data[:, :, wind_direction_index] = temporal_data[:, :, wind_direction_index] / 360.0

        return normalized_temporal_data

    def compute_utci_global_max_min(self):
        for i, output_file in enumerate(self.output_data_paths):
            output_data = self.output_data[i]
            output_data_tensor = torch.tensor(output_data).unsqueeze(0)
            current_max = output_data_tensor.max().item()
            current_min = output_data_tensor.min().item()

            if self.utci_max is None or current_max > self.utci_max:
                self.utci_max = current_max
            if self.utci_min is None or current_min < self.utci_min:
                self.utci_min = current_min

        print(f"全局 UTCI 最大值: {self.utci_max}, 最小值: {self.utci_min}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        h_idx = random.randint(0,3917)
        w_idx = random.randint(0,3674)
        sample_idx = idx // (336 - (self.T_in + self.T_out - 1))
        time_idx = idx % (336 - (self.T_in + self.T_out - 1))
        spatial_data_seq = TF.crop(self.spatial_data, top=h_idx, left=w_idx, height=64, width=64).cpu()
        spatial_data_seq = spatial_data_seq.unsqueeze(-1).repeat(1, 1, 1, self.T_in)  # [4, H, W, T_in]
        spatial_data_seq = spatial_data_seq.permute(1, 2, 0, 3)  # [H, W, 4, T_in]

        temporal_data = torch.tensor(self.temporal_data[sample_idx, time_idx:time_idx + self.T_in], dtype=torch.float32).cpu()

        output_data_list = []
        for t in range(self.T_out):
            output_data = self.output_data[time_idx + self.T_in + t]
            output_data = torch.tensor(output_data).unsqueeze(0)
            output_data, _, _ = self.maxminscaler_3d(output_data, self.utci_max, self.utci_min)
            output_data = TF.crop(output_data, top=h_idx, left=w_idx, height=64, width=64).cpu()
            output_data_list.append(output_data)
        output_data = torch.stack(output_data_list).permute(2, 3, 1, 0)
        utci_input_list = []
        for t in range(self.T_in):
            utci_data = self.output_data[time_idx + t]
            utci_data = torch.tensor(utci_data).unsqueeze(0)
            utci_data, _, _ = self.maxminscaler_3d(utci_data, self.utci_max, self.utci_min)
            utci_data = TF.crop(utci_data, top=h_idx, left=w_idx, height=64, width=64).cpu()
            utci_input_list.append(utci_data)

        utci_input = torch.stack(utci_input_list).permute(2, 3, 1, 0)  
        spatial_data_seq = torch.cat([spatial_data_seq, utci_input], dim=2) 

        return spatial_data_seq.cpu(), output_data.cpu(), temporal_data.cpu()

    def maxminscaler_3d(self, tensor_3d, scaler_max=None, scaler_min=None, range=(0, 1)):
        if scaler_max is None:
            scaler_max = tensor_3d.max()
        if scaler_min is None:
            scaler_min = tensor_3d.min()
        X_std = (tensor_3d - scaler_min) / (scaler_max - scaler_min)
        X_scaled = X_std * (range[1] - range[0]) + range[0]
        return X_scaled, scaler_max, scaler_min
