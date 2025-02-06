import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
import os
import tifffile
import csv
from dataloader import CustomDataset
from Transformer import SpatialTemporalTransformer_Decoder

def inverse_scaler(tensor, scaler_max, scaler_min):
    return tensor * (scaler_max - scaler_min) + scaler_min

def save_predictions_to_csv(predictions, ground_truths, output_file, num_nodes, T_out):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        for node_id in range(num_nodes):  
            row = [f"{node_id+1}"]  
            for t in range(T_out):  
                pred = predictions[node_id, t]
                gt = ground_truths[node_id, t]
                row.append(f"{pred},{gt}")  
            writer.writerow(row)

def predict_and_save(model, dataloader, output_file, num_nodes, T_out, device='cuda'):
    model.eval()
    predictions_list = np.zeros((num_nodes, T_out))  
    ground_truth_list = np.zeros((num_nodes, T_out))  
    utci_max = dataloader.dataset.dataset.utci_max  
    utci_min = dataloader.dataset.dataset.utci_min  
    print(f"Max UTCI: {utci_max}, Min UTCI: {utci_min}")  
    node_counter = 0  

    with torch.no_grad():
        for spatial_data, output_data, temporal_data in dataloader:
            spatial_data = spatial_data.to(device)
            output_data = output_data.to(device)
            temporal_data = temporal_data.to(device)
            print(np.shape(spatial_data), np.shape(temporal_data))
            predictions = model(spatial_data, temporal_data)
            predictions_np = predictions.cpu().detach().numpy()
            output_data_np = output_data.cpu().detach().numpy()
            print(f"Predictions shape: {predictions_np.shape}")
            print(f"Output data shape before transpose: {output_data_np.shape}")
            predictions_np = np.transpose(predictions_np, (0, 4, 1, 2, 3))

            if len(output_data_np.shape) == 5 and output_data_np.shape[3] == 1:
                output_data_np = np.transpose(output_data_np, (0, 4, 1, 2, 3))  

            if predictions_np.shape[2] == 1:
                predictions_np = np.squeeze(predictions_np, axis=2)  

            if output_data_np.shape[4] == 1:
                output_data_np = np.squeeze(output_data_np, axis=4)  

            batch_size = predictions_np.shape[0] 

            for i in range(batch_size):  
                for h in range(64): 
                    for w in range(64):  
                        node_idx = node_counter + h * 64 + w  
                        if node_idx < num_nodes:
                            for t in range(T_out):  
                                scalar_prediction = float(predictions_np[i, t, h, w])
                                scalar_ground_truth = float(output_data_np[i, t, h, w])  

                                predictions_list[node_idx, t] = inverse_scaler(scalar_prediction, utci_max, utci_min)
                                ground_truth_list[node_idx, t] = inverse_scaler(scalar_ground_truth, utci_max, utci_min)

            node_counter += 64 * 64  
    print(np.shape(ground_truth_list))
    save_predictions_to_csv(predictions_list, ground_truth_list, output_file, num_nodes, T_out)

def split_dataset_by_time(spatial_data_dir, temporal_data_csv, output_data_dir, train_ratio=0.4, val_ratio=0.3):
    dataset = CustomDataset(spatial_data_dir, temporal_data_csv, output_data_dir, T_in=24, T_out=24)
    total_samples = len(dataset) 
    train_size = int(train_ratio * total_samples)  
    val_size = int(val_ratio * total_samples)  
    test_size = total_samples - train_size - val_size  
    val_indices = list(range(train_size, train_size + val_size))
    val_dataset = Subset(dataset, val_indices)
    return val_dataset

spatial_data_dir = '/root/UTCI_prediction/data/spatial_images'
temporal_data_csv = '/root/UTCI_prediction/data/weather data.csv'
output_data_dir = '/root/UTCI_prediction/data/output_images'

test_dataset = split_dataset_by_time(spatial_data_dir, temporal_data_csv, output_data_dir)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

H = 64
W = 64
C_in = 5
T_in = 24
C_temp = 7
T_out = 24
C_out = 1
model = SpatialTemporalTransformer_Decoder(H=H, W=W, C_in=C_in, C_temp=C_temp, C_out=C_out,
                                   T_in=T_in, hidden_dim=12, num_heads=2,
                                   num_layers=1, dropout=0.1)

checkpoint = torch.load('output/1010_model_trained_40.pth')
# model = checkpoint.module
new_state_dict = {}
for k,v in checkpoint.items():
    new_state_dict[k[7:]] = v
model.load_state_dict(new_state_dict)
model.to('cuda')
num_nodes = 64 * 64 * len(test_loader)  
T_out = 24  
output_file = './1010_test_predictions.csv'
predict_and_save(model, test_loader, output_file, num_nodes, T_out)
