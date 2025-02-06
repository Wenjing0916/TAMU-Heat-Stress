import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  
import numpy as np
from dataloader import CustomDataset
import gc
from Transformer import SpatialTemporalTransformer_Decoder
from utils.utils import replace_w_sync_bn, CustomDataParallel
gc.collect()
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def split_dataset_by_time(spatial_data_dir, temporal_data_csv, output_data_dir, train_ratio=0.7, val_ratio=0.15):
    dataset = CustomDataset(spatial_data_dir, temporal_data_csv, output_data_dir, T_in=24, T_out=24)
    total_samples = len(dataset) 
    train_size = int(train_ratio * total_samples) 
    val_size = int(val_ratio * total_samples)  
    test_size = total_samples - train_size - val_size 

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_and_evaluate_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.0001, log_dir='logs/1010_model', patience=10, min_delta=0.0005):
    writer = SummaryWriter(log_dir)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    
    model.apply(replace_w_sync_bn)
    model = CustomDataParallel(model, 2)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (spatial_data, output_data, temporal_data) in enumerate(tqdm(train_loader)):
            spatial_data = spatial_data.to(device,non_blocking=True)
            output_data = output_data.to(device,non_blocking=True)
            temporal_data = temporal_data.to(device,non_blocking=True)  
            optimizer.zero_grad()
            outputs = model(spatial_data, temporal_data)
            loss = criterion(outputs, output_data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), './output/1010_model_trained_{}.pth'.format(epoch))
            model.eval()  
            val_loss = 0.0
            with torch.no_grad():  
                for spatial_data_val, output_data_val, temporal_data_val in val_loader:
                    spatial_data_val = spatial_data_val.to(device,non_blocking=True)
                    output_data_val = output_data_val.to(device,non_blocking=True)
                    temporal_data_val = temporal_data_val.to(device,non_blocking=True)  
                    outputs_val = model(spatial_data_val, temporal_data_val)
                    loss_val = criterion(outputs_val, output_data_val)
                    val_loss += loss_val.item()

            avg_val_loss = val_loss / len(val_loader)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    writer.close()

    torch.save(model.module.state_dict(), '/root/UTCI_prediction/code/output1/1010_model_trained.pth')
    print("Training complete, model saved.")

   
    def calculate_metrics(loader, dataset_type):
        model.eval()  
        total_loss = 0.0
        mse_criterion = nn.MSELoss()

        with torch.no_grad():  
            for spatial_data, output_data, temporal_data in loader:
                spatial_data = spatial_data.to(device,non_blocking=True)
                output_data = output_data.to(device,non_blocking=True)
                temporal_data = temporal_data.to(device,non_blocking=True)  
                outputs = model(spatial_data, temporal_data)
                loss = mse_criterion(outputs, output_data)
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        rmse = np.sqrt(avg_loss)
        print(f"{dataset_type} - MSE: {avg_loss:.4f}, RMSE: {rmse:.4f}")

    calculate_metrics(train_loader, "Training Set")
    calculate_metrics(val_loader, "Validation Set")


spatial_data_dir = '/root/UTCI_prediction/data/spatial_images'
temporal_data_csv = '/root/UTCI_prediction/data/weather data.csv'
output_data_dir = '/root/UTCI_prediction/data/output_images'

train_dataset, val_dataset = split_dataset_by_time(spatial_data_dir, temporal_data_csv, output_data_dir)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, pin_memory=True, num_workers=10, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, pin_memory=True, num_workers=10, drop_last=True)

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

train_and_evaluate_model(model, train_loader, val_loader, num_epochs=200, learning_rate=0.0001)
