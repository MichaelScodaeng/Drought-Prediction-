# --- CNN-LSTM Model Definition ---
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from src.data_utils import load_and_prepare_data, split_data_chronologically,load_config
from src.preprocess_utils import scale_data
from src.grid_utils import create_gridded_data
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,root_mean_squared_error
import json
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import torch.nn.functional as F
import optuna

class CNNLSTMModel(nn.Module):
    def __init__(self, in_channels, hidden_cnn=16, kernel_size=3, lstm_hidden_size=64,
                 lstm_layers=1, dropout=0.0, height=20, width=20, use_layernorm=False):
        super().__init__()
        padding = kernel_size // 2

        # CNN encoder for each time slice
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_cnn, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(hidden_cnn),
            nn.ReLU(),
            nn.Conv2d(hidden_cnn, hidden_cnn, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(hidden_cnn),
            nn.ReLU(),
        )

        self.height = height
        self.width = width
        self.hidden_cnn = hidden_cnn
        self.use_layernorm = use_layernorm

        # Flattened CNN output size for LSTM input
        self.flattened_size = hidden_cnn * height * width

        self.lstm = nn.LSTM(
            input_size=self.flattened_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        if use_layernorm:
            self.ln = nn.LayerNorm(lstm_hidden_size)

        # Output layer: map LSTM output back to spatial grid
        self.fc = nn.Linear(lstm_hidden_size, height * width)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        cnn_out = []
        for t in range(T):
            xt = x[:, t]  # [B, C, H, W]
            feat = self.cnn_encoder(xt)  # [B, hidden_cnn, H, W]
            feat_flat = feat.reshape(B, -1)  # [B, hidden_cnn * H * W]
            cnn_out.append(feat_flat)

        cnn_seq = torch.stack(cnn_out, dim=1)  # [B, T, hidden_cnn * H * W]
        lstm_out, _ = self.lstm(cnn_seq)       # [B, T, lstm_hidden_size]

        last_out = lstm_out[:, -1, :]          # [B, lstm_hidden_size]
        if self.use_layernorm:
            last_out = self.ln(last_out)

        pred = self.fc(last_out)               # [B, H * W]
        pred = pred.view(B, H, W)              # [B, H, W]
        return pred
class GriddedCNNLSTMDataset(Dataset):
    def __init__(self, gridded_data, target_variable, feature_names, n_steps_in):
        self.data = torch.tensor(gridded_data, dtype=torch.float32)  # [T, H, W, F]
        self.target_idx = feature_names.index(target_variable)
        self.feature_names = feature_names
        self.n_steps_in = n_steps_in

    def __len__(self):
        return self.data.shape[0] - self.n_steps_in

    def __getitem__(self, idx):
        # Extract sequence
        X_seq = self.data[idx:idx + self.n_steps_in]           # [T, H, W, F]
        y = self.data[idx + self.n_steps_in, :, :, self.target_idx]  # [H, W]

        # Rearrange to [T, C, H, W]
        X_seq = X_seq.permute(0, 3, 1, 2)  # [T, C, H, W]
        return X_seq, y
# --- Pipeline Skeleton for CNN-LSTM ---
class CNNLSTMPipeline:
    def _get_abs_path_from_config_value(self, relative_path):
        if not relative_path or os.path.isabs(relative_path):
            return relative_path
        return os.path.abspath(os.path.join(self.project_root_for_paths, relative_path))

    def __init__(self, config_path):
        self.cfg = load_config(config_path)
        self.config_path_abs = os.path.abspath(config_path)
        self.project_root_for_paths = os.path.join(os.path.dirname(self.config_path_abs), "..", "..")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.raw_path = self._get_abs_path_from_config_value(self.cfg.get('data', {}).get('raw_data_path'))
        self.experiment_name = self.cfg.get('project_setup', {}).get('experiment_name', 'cnnlstm_experiment')
        self.run_output_dir = os.path.join(self.project_root_for_paths, 'run_outputs', self.experiment_name)
        self.run_models_dir = os.path.join(self.project_root_for_paths, 'models_saved', self.experiment_name)
        self.best_model_path = os.path.join(self.run_models_dir, 'cnnlstm_best_model.pt')
        self.final_model_path = os.path.join(self.run_models_dir, 'cnnlstm_final_model.pt')

        os.makedirs(self.run_output_dir, exist_ok=True)
        os.makedirs(self.run_models_dir, exist_ok=True)

        self._prepare_data()

    def _prepare_data(self):
        full_df = load_and_prepare_data({
            'data': {
                'raw_data_path': self.raw_path,
                'time_column': self.cfg['data']['time_column']
            }
        })
        featured_df = full_df.copy()
        train_df, val_df, test_df = split_data_chronologically(featured_df, self.cfg)
        scaled_train, scaled_val, scaled_test, scaler = scale_data(train_df, val_df, test_df, self.cfg)
        self.scaler = scaler
        self.target_scaler = RobustScaler().fit(train_df[self.cfg['project_setup']['target_variable']].values.reshape(-1, 1))

        target_col = self.cfg['project_setup']['target_variable']
        feature_names = self.cfg['data']['features_to_grid']
        n_steps_in = self.cfg['cnnlstm_params']['n_steps_in']

        self.gridded_data_train, self.mask_train = create_gridded_data(scaled_train, self.cfg)
        self.gridded_data_val, self.mask_val = create_gridded_data(scaled_val, self.cfg)
        self.gridded_data_test, self.mask_test = create_gridded_data(scaled_test, self.cfg)
        self.mask = self.mask_train

        self.target_variable = target_col
        self.feature_names = feature_names
        self.n_steps_in = n_steps_in

        self.train_dataset = GriddedCNNLSTMDataset(self.gridded_data_train, target_col, feature_names, n_steps_in)
        self.val_dataset = GriddedCNNLSTMDataset(self.gridded_data_val, target_col, feature_names, n_steps_in)
        self.test_dataset = GriddedCNNLSTMDataset(self.gridded_data_test, target_col, feature_names, n_steps_in)

    def train(self):
        params = self.cfg['cnnlstm_params']
        batch_size = params.get('batch_size', 4)
        n_epochs = params.get('max_epochs', 10)
        lr = params.get('learning_rate', 1e-3)
        patience = params.get('patience', 10)
        hidden_cnn = params.get('hidden_cnn', 16)
        kernel_size = params.get('kernel_size', 3)
        lstm_hidden_size = params.get('lstm_hidden_size', 64)
        lstm_layers = params.get('lstm_layers', 1)
        dropout = params.get('dropout', 0.0)
        use_layernorm = params.get('use_layernorm', False)

        writer = SummaryWriter(log_dir=os.path.join(self.run_output_dir, 'tensorboard_logs'))

        model = CNNLSTMModel(
            in_channels=len(self.feature_names),
            hidden_cnn=hidden_cnn,
            kernel_size=kernel_size,
            lstm_hidden_size=lstm_hidden_size,
            lstm_layers=lstm_layers,
            dropout=dropout,
            use_layernorm=use_layernorm,
            height=self.gridded_data_train.shape[1],
            width=self.gridded_data_train.shape[2]
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size)

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(n_epochs):
            model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.float().to(self.device), yb.float().to(self.device)
                optimizer.zero_grad()
                out = model(Xb)
                mask = torch.tensor(self.mask_train, device=self.device).unsqueeze(0).expand(out.shape[0], -1, -1)
                loss = criterion(out * mask, yb * mask)
                loss.backward()
                optimizer.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.float().to(self.device), yb.float().to(self.device)
                    out = model(Xb)
                    mask = torch.tensor(self.mask_val, device=self.device).unsqueeze(0).expand(out.shape[0], -1, -1)
                    val_loss = criterion(out * mask, yb * mask)
                    val_losses.append(val_loss.item())

            mean_val_loss = np.mean(val_losses)
            print(f"Epoch {epoch+1}/{n_epochs} - Val Loss: {mean_val_loss:.4f}")
            writer.add_scalar("Loss/val", mean_val_loss, epoch)
            writer.add_scalar("Loss/train", loss.item(), epoch)

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), self.best_model_path)
                print(f"✅ New best model saved to {self.best_model_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"🛑 Early stopping triggered after {epoch+1} epochs.")
                    break

        self.model = model
        writer.close()

    # (The existing content remains unchanged above this point)

    # (The existing content remains unchanged above this point)

    def evaluate(self):
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()

        loaders = {
            'train': DataLoader(self.train_dataset, batch_size=8, shuffle=False),
            'val': DataLoader(self.val_dataset, batch_size=8, shuffle=False),
            'test': DataLoader(self.test_dataset, batch_size=8, shuffle=False)
        }

        metrics = {}

        for split, loader in loaders.items():
            preds, actuals = [], []
            with torch.no_grad():
                for Xb, yb in loader:
                    Xb = Xb.float().to(self.device)
                    out = self.model(Xb).cpu().numpy()
                    preds.append(out)
                    actuals.append(yb.cpu().numpy())

            preds = np.concatenate(preds, axis=0)  # [B, H, W]
            actuals = np.concatenate(actuals, axis=0)  # [B, H, W]

            mask = self.mask_train if split == 'train' else self.mask_val if split == 'val' else self.mask_test
            valid_mask = mask.astype(bool)

            preds_valid = preds[:, valid_mask]  # [B, valid_pixels]
            actuals_valid = actuals[:, valid_mask]  # [B, valid_pixels]

            preds_valid = preds_valid.reshape(-1, 1)
            actuals_valid = actuals_valid.reshape(-1, 1)

            preds_inv = self.target_scaler.inverse_transform(preds_valid)
            actuals_inv = self.target_scaler.inverse_transform(actuals_valid)

            rmse = float(root_mean_squared_error(actuals_inv, preds_inv))
            mae = float(mean_absolute_error(actuals_inv, preds_inv))
            r2 = float(r2_score(actuals_inv, preds_inv))

            metrics[split] = {'rmse': rmse, 'mae': mae, 'r2': r2}
            print(f"{split.capitalize()} Set - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        metrics_filename = os.path.join(self.run_output_dir, f'evaluation_metrics.json')
        with open(metrics_filename, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_filename}")
        return metrics




    def predict_on_full_data(self):
        from tqdm import tqdm
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()

        full_dataset = torch.utils.data.ConcatDataset([
            self.train_dataset,
            self.val_dataset,
            self.test_dataset
        ])
        full_loader = DataLoader(full_dataset, batch_size=8, shuffle=False)

        preds = []
        with torch.no_grad():
            for Xb, _ in full_loader:
                Xb = Xb.float().to(self.device)
                out = self.model(Xb).squeeze().cpu().numpy()
                preds.append(out)

        predicted_grids = np.concatenate(preds)  # [T, H, W]

        full_df_raw = load_and_prepare_data({
            'data': {
                'raw_data_path': self.raw_path,
                'time_column': self.cfg['data']['time_column']
            }
        })

        lat_col = self.cfg['data']['lat_column']
        lon_col = self.cfg['data']['lon_column']
        time_col = self.cfg['data']['time_column']
        fixed_step = self.cfg.get('gridding', {}).get('fixed_step', 0.5)
        lat_min = full_df_raw[lat_col].min()
        lon_min = full_df_raw[lon_col].min()

        full_df_raw['row_idx'] = ((full_df_raw[lat_col] - lat_min) / fixed_step).round().astype(int)
        full_df_raw['col_idx'] = ((full_df_raw[lon_col] - lon_min) / fixed_step).round().astype(int)

        indexed_df = full_df_raw.set_index([time_col, 'row_idx', 'col_idx'])
        time_steps = np.sort(full_df_raw[time_col].unique())
        prediction_times = time_steps[self.n_steps_in:self.n_steps_in + len(predicted_grids)]

        coord_cols = ['row_idx', 'col_idx', lat_col, lon_col]
        cell_to_coord = full_df_raw[coord_cols].drop_duplicates().set_index(['row_idx', 'col_idx'])

        valid_pixel_indices = np.argwhere(self.mask == 1)
        target = self.cfg['project_setup']['target_variable']
        output_records = []

        for i, pred_time in enumerate(tqdm(prediction_times, desc="Saving predictions")):
            if i >= len(predicted_grids):
                break
            pred_grid = predicted_grids[i]  # [H, W]
            for r, c in valid_pixel_indices:
                try:
                    actual_val = indexed_df.loc[(pred_time, r, c), target]
                except KeyError:
                    actual_val = np.nan
                try:
                    coords = cell_to_coord.loc[(r, c)]
                    lat = coords[lat_col]
                    lon = coords[lon_col]
                except KeyError:
                    continue

                pred_val = pred_grid[r, c]
                pred_val = self.target_scaler.inverse_transform([[pred_val]])[0][0]

                output_records.append({
                    'time': pred_time,
                    'lat': lat,
                    'lon': lon,
                    target: actual_val,
                    f'{target}_predicted': pred_val,
                    f'{target}_prediction_error': pred_val - actual_val if not np.isnan(actual_val) else np.nan,
                    f'{target}_absolute_error': abs(pred_val - actual_val) if not np.isnan(actual_val) else np.nan
                })

        output_df = pd.DataFrame(output_records)
        output_path = os.path.join(self.run_output_dir, 'full_data_predictions.csv')
        output_df.to_csv(output_path, index=False)
        print(f"✅ Full predictions saved to: {output_path}")
        return output_df
    
    def objective(self, trial):
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
        hidden_cnn = trial.suggest_categorical('hidden_cnn', [8, 16, 32])
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
        lstm_hidden_size = trial.suggest_categorical('lstm_hidden_size', [32, 64, 128])
        lstm_layers = trial.suggest_int('lstm_layers', 1, 2)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        lr = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

        model = CNNLSTMModel(
            in_channels=len(self.feature_names),
            hidden_cnn=hidden_cnn,
            kernel_size=kernel_size,
            lstm_hidden_size=lstm_hidden_size,
            lstm_layers=lstm_layers,
            dropout=dropout,
            use_layernorm=False,
            height=self.gridded_data_train.shape[1],
            width=self.gridded_data_train.shape[2]
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size)

        for epoch in range(self.cfg['cnnlstm_params'].get('max_epochs_tune', 5)):
            model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.float().to(self.device), yb.float().to(self.device)
                optimizer.zero_grad()
                out = model(Xb)
                mask = torch.tensor(self.mask_train, device=self.device).unsqueeze(0).expand(out.shape[0], -1, -1)
                loss = criterion(out * mask, yb * mask)
                loss.backward()
                optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.float().to(self.device), yb.float().to(self.device)
                out = model(Xb)
                mask = torch.tensor(self.mask_val, device=self.device).unsqueeze(0).expand(out.shape[0], -1, -1)
                val_loss = criterion(out * mask, yb * mask)
                val_losses.append(val_loss.item())

        return np.mean(val_losses)

    def tune_and_train(self, n_trials=10):
        import optuna
        study = optuna.create_study(direction='minimize')
        n_trials = self.cfg['optuna_tuning'].get('n_trials', n_trials)
        study.optimize(self.objective, n_trials=n_trials)
        best_params = study.best_trial.params
        print("Best trial:", best_params)

        # Merge with fixed params
        config_params = self.cfg['cnnlstm_params']
        final_params = {
            'batch_size': best_params.get('batch_size', config_params.get('batch_size', 8)),
            'hidden_cnn': best_params.get('hidden_cnn', config_params.get('hidden_cnn', 16)),
            'kernel_size': best_params.get('kernel_size', config_params.get('kernel_size', 3)),
            'lstm_hidden_size': best_params.get('lstm_hidden_size', config_params.get('lstm_hidden_size', 64)),
            'lstm_layers': best_params.get('lstm_layers', config_params.get('lstm_layers', 1)),
            'dropout': best_params.get('dropout', config_params.get('dropout', 0.0)),
            'learning_rate': best_params.get('learning_rate', config_params.get('learning_rate', 1e-3)),
            'use_layernorm': config_params.get('use_layernorm', False),
            'max_epochs': config_params.get('max_epochs', 10),
            'patience': config_params.get('patience', 10)
        }

        self.cfg['cnnlstm_params'].update(final_params)

        with open(os.path.join(self.run_output_dir, 'best_hyperparams.json'), 'w') as f:
            json.dump(final_params, f, indent=4)

        print("Retraining final model with best params...")
        self.train()
        torch.save(self.model.state_dict(), self.final_model_path)
        print(f"Final model saved to {self.final_model_path}")

