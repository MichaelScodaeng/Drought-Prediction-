
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.preprocess_utils import scale_data, inverse_transform_predictions
from src.feature_utils import engineer_features
from src.data_utils import load_and_prepare_data, split_data_chronologically
from src.cam_clstm.causal_clsm_model import MyConvLSTMModel
import torch
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset

class GriddedConvLSTMDataset(Dataset):
    def __init__(self, X, y_dict, temporal_only=None):
        self.X = X
        self.y_dict = y_dict
        self.temporal_only = temporal_only  # torch.Tensor [B, T, C_temp]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = {k: v[idx] for k, v in self.y_dict.items()}
        if self.temporal_only is not None:
            t_only = self.temporal_only[idx]
            return {'x': x, 'y': y, 'temporal_only': t_only}
        else:
            return {'x': x, 'y': y}


class GriddedConvLSTMDatasetWithMeta(Dataset):
    def __init__(self, X, Y_dict, times, lats, lons):
        """
        X: np.ndarray [N, T, C, H, W]
        Y_dict: dict of [N, 1, H, W]
        times: list-like [N] — timestamp of each sample
        lats, lons: 1D arrays for spatial grid [H], [W]
        """
        self.X = torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X
        self.Y_dict = {
            k: torch.tensor(v, dtype=torch.float32) if not torch.is_tensor(v) else v
            for k, v in Y_dict.items()
        }
        self.times = times  # list of datetime or string
        self.lats = torch.tensor(lats, dtype=torch.float32)
        self.lons = torch.tensor(lons, dtype=torch.float32)
        self.N, _, _, H, W = self.X.shape
        self.task_keys = list(self.Y_dict.keys())

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = self.X[idx]  # [T, C, H, W]
        y = {k: self.Y_dict[k][idx] for k in self.task_keys}  # [1, H, W] per task
        meta = {
            'time': self.times[idx],  # e.g., datetime object
            'lat': self.lats.view(-1),  # [H]
            'lon': self.lons.view(-1)   # [W]
        }
        return x, y, meta


def load_causal_masks(feature_names):
    masks = {
        'pre': torch.ones(len(feature_names)),
        'pet': torch.ones(len(feature_names)),
        'spei': torch.ones(len(feature_names))
    }
    return masks

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        x, y_dict = batch
        x = x.to(device)
        y_dict = {k: v.to(device) for k, v in y_dict.items()}
        optimizer.zero_grad()
        preds, loss, losses = model(x, y_true=y_dict)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_trues = {'pre': [], 'pet': [], 'spei': []}, {'pre': [], 'pet': [], 'spei': []}
    with torch.no_grad():
        for x, y_dict in dataloader:
            x = x.to(device)
            y_dict = {k: v.to(device) for k, v in y_dict.items()}
            preds = model(x)
            for key in ['pre', 'pet', 'spei']:
                all_preds[key].append(preds[key].cpu().numpy())
                all_trues[key].append(y_dict[key].cpu().numpy())

    metrics = {}
    for key in ['pre', 'pet', 'spei']:
        y_pred = np.concatenate(all_preds[key])
        y_true = np.concatenate(all_trues[key])
        metrics[key] = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    return metrics

def run_pipeline(config):
    df = load_and_prepare_data(config)
    df = engineer_features(df, config)
    df_train, df_val, df_test = split_data_chronologically(df, config)

    (X_train, y_train_dict), scaler_x, scaler_y = scale_data(df_train, return_dict=True)
    (X_val, y_val_dict), _, _ = scale_data(df_val, scaler_x, scaler_y, return_dict=True)
    (X_test, y_test_dict), _, _ = scale_data(df_test, scaler_x, scaler_y, return_dict=True)

    dataset_train = GriddedConvLSTMDatasetWithMeta(X_train, y_train_dict)
    dataset_val = GriddedConvLSTMDatasetWithMeta(X_val, y_val_dict)
    train_loader = DataLoader(dataset_train, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=8)

    causal_masks = load_causal_masks(config['feature_names'])

    model = MyConvLSTMModel(
        input_channels=len(config['feature_names']),
        height=config['grid']['height'],
        width=config['grid']['width'],
        hidden_channels=32,
        use_pos_enc=True,
        use_spatial_attn=True,
        use_temporal_only=True,
        causal_masks=causal_masks
    ).to(config['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(config['training']['epochs']):
        train_loss = train(model, train_loader, optimizer, config['device'])
        val_metrics = evaluate(model, val_loader, config['device'])
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Metrics: {val_metrics}")
