import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.data_utils import load_config, load_and_prepare_data, split_data_chronologically
from src.preprocess_utils import scale_data, inverse_transform_predictions
import joblib
from src.feature_utils import engineer_features
import json
import optuna
from torch.utils.tensorboard import SummaryWriter
from src.grid_utils import create_gridded_data
# --- CNN3D Model Definition --- 
class CNN3DModel(nn.Module):
    def __init__(self, in_channels, out_channels, n_conv_layers=2, hidden_channels=16,
                 kernel_size=(3, 3, 3), use_batchnorm=False, dropout_rate=0.0, n_steps_in=12):
        super().__init__()
        layers = []
        for i in range(n_conv_layers):
            layers.append(nn.Conv3d(
                in_channels if i == 0 else hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=tuple(k // 2 for k in kernel_size)
            ))
            if use_batchnorm:
                layers.append(nn.BatchNorm3d(hidden_channels))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout3d(dropout_rate))

        # Final conv collapses time dimension
        layers.append(nn.Conv3d(hidden_channels, out_channels, kernel_size=(n_steps_in, 1, 1)))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)           # [B, 1, 1, H, W]
        return x.squeeze(2)       # → [B, 1, H, W]

# --- Dataset for CNN3D ---
from torch.utils.data import Dataset
    
class GriddedCNN3DDataset(Dataset):
    def __init__(self, gridded_data, target_variable, feature_names, n_steps_in):
        self.data = torch.tensor(gridded_data, dtype=torch.float32)
        self.target_idx = feature_names.index(target_variable)
        self.n_steps_in = n_steps_in

    def __len__(self):
        return self.data.shape[0] - self.n_steps_in

    def __getitem__(self, idx):
        X = self.data[idx:idx + self.n_steps_in]                         # [D, H, W, F]
        y = self.data[idx + self.n_steps_in, :, :, self.target_idx]     # [H, W]
        X = X.permute(3, 0, 1, 2)                                        # [F, D, H, W]
        return X, y

def masked_mse(pred, target, mask):
    valid = mask.bool()
    return ((pred - target)[valid] ** 2).mean()

# --- Pipeline ---
class CNN3DPipeline:
    def _get_abs_path_from_config_value(self, relative_path): # (Helper function)
        if not relative_path or os.path.isabs(relative_path): return relative_path
        return os.path.abspath(os.path.join(self.project_root_for_paths, relative_path))
    def __init__(self, config_path):
        self.cfg = load_config(config_path)
        self.config_path_abs = os.path.abspath(config_path)
        self.project_root_for_paths = os.path.join(os.path.dirname(self.config_path_abs),"..","..")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.raw_path = self._get_abs_path_from_config_value(self.cfg.get('data', {}).get('raw_data_path'))
        self.experiment_name = self.cfg.get('project_setup', {}).get('experiment_name', 'cnn3d_experiment')
        self.run_output_dir = os.path.join(self.project_root_for_paths, 'run_outputs', self.experiment_name)
        self.run_models_dir = os.path.join(self.project_root_for_paths, 'models_saved', self.experiment_name)
        self.best_model_path = os.path.join(self.run_models_dir, 'cnn3d_best_model.pt')
        self.final_model_path = os.path.join(self.run_models_dir, 'cnn3d_final_model.pt')
        
        #self.scaler_path = self._get_abs_path_from_config_value(self.cfg.get('data', {}).get('scaler_path'))
        #self.target_scaler = joblib.load(self.scaler_path)
        self._prepare_data()
        
        
        
        
        
        os.makedirs(self.run_output_dir, exist_ok=True)
        os.makedirs(self.run_models_dir, exist_ok=True)

    def _prepare_data(self):
        # Load and feature engineer
        full_df = load_and_prepare_data({
            'data': {
                'raw_data_path': self.raw_path,
                'time_column': self.cfg['data']['time_column']
            }
        })
        featured_df = full_df.copy()
        train_df, val_df, test_df = split_data_chronologically(featured_df, self.cfg)
        # Scaling
        scaled_train, scaled_val, scaled_test, scaler = scale_data(train_df, val_df, test_df, self.cfg)
        self.scaler = scaler
        target_col = self.cfg['project_setup']['target_variable']
        feature_names = self.cfg['data']['features_to_grid']
        n_steps_in = self.cfg['cnn3d_params']['n_steps_in']
        self.gridded_data_train, self.mask_train = create_gridded_data(scaled_train, self.cfg)
        self.gridded_data_val, self.mask_val = create_gridded_data(scaled_val, self.cfg)
        self.gridded_data_test, self.mask_test = create_gridded_data(scaled_test, self.cfg)
        self.mask = self.mask_train
        self.target_variable = target_col
        self.feature_names = feature_names
        self.n_steps_in = n_steps_in
        # Datasets
        self.train_dataset = GriddedCNN3DDataset(self.gridded_data_train, target_col, feature_names, n_steps_in)
        self.val_dataset = GriddedCNN3DDataset(self.gridded_data_val, target_col, feature_names, n_steps_in)
        self.test_dataset = GriddedCNN3DDataset(self.gridded_data_test, target_col, feature_names, n_steps_in)

    def train(self):
        params = self.cfg['cnn3d_params']
        batch_size = params.get('batch_size', 4)
        n_conv_layers = params.get('n_conv_layers', 2)
        hidden_channels = params.get('hidden_channels', 16)
        kernel_size = tuple(params.get('kernel_size', [3, 3, 3]))
        n_epochs = params.get('max_epochs', 10)
        lr = params.get('learning_rate', 1e-3)
        patience = params.get('patience', 10)
        use_batchnorm = params.get('use_batchnorm', False)
        dropout_rate = params.get('dropout', 0.0)
        writer = SummaryWriter(log_dir=os.path.join(self.run_output_dir, 'tensorboard_logs'))
        model = CNN3DModel(
            in_channels=len(self.feature_names),
            out_channels=1,
            n_conv_layers=n_conv_layers,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            use_batchnorm=use_batchnorm,
            dropout_rate=dropout_rate
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size)

        best_val_loss = float('inf')
        epochs_no_improve = 0

        # Save hyperparameters
        hparams = {
            'batch_size': batch_size,
            'n_conv_layers': n_conv_layers,
            'hidden_channels': hidden_channels,
            'kernel_size': kernel_size,
            'learning_rate': lr,
            'dropout': dropout_rate,
            'use_batchnorm': use_batchnorm,
            'max_epochs': n_epochs,
            'patience': patience
        }
        with open(self.best_model_path.replace('.pt', '_hparams.json'), 'w') as f:
            json.dump(hparams, f, indent=4)

        for epoch in range(n_epochs):
            model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.float().to(self.device), yb.float().to(self.device)
                optimizer.zero_grad()
                out = model(Xb).squeeze(1)
                mask = torch.tensor(self.mask_train, device=self.device)
                mask = mask.unsqueeze(0).expand(out.shape[0], -1, -1)  # match batch size
                loss = criterion(out * mask, yb.squeeze() * mask)
                loss.backward()
                optimizer.step()
            best_model_path =self.best_model_path
            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.float().to(self.device), yb.float().to(self.device)
                    out = model(Xb).squeeze(1)
                    mask = torch.tensor(self.mask_val, device=self.device)  # [H, W]
                    mask = mask.unsqueeze(0).expand(out.shape[0], -1, -1)   # [B, H, W]
                    val_loss = masked_mse(out, yb.squeeze(), mask)

                    val_losses.append(val_loss.item())
            mean_val_loss = np.mean(val_losses)
            print(f"Epoch {epoch+1}/{n_epochs} - Val Loss: {mean_val_loss:.4f}")

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"✅ New best model saved to {best_model_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"🛑 Early stopping triggered after {epoch+1} epochs.")
                    break

        self.model = model


    def evaluate(self):
        self.model.load_state_dict(torch.load(self.final_model_path))
        self.model.eval()
        train_loader = DataLoader(self.train_dataset, batch_size=8, shuffle=False)
        val_loader = DataLoader(self.val_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=8, shuffle=False)
        loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
        #save metrics
        metrics = {}

        for split, loader in loaders.items():
            print(f"Evaluating on {split} set...")
            preds, actuals = [], []
            with torch.no_grad():
                for Xb, yb in loader:
                    Xb = Xb.float().to(self.device)
                    out = self.model(Xb).squeeze().cpu().numpy()
                    preds.append(out)
                    actuals.append(yb.squeeze().cpu().numpy())
            preds = np.concatenate(preds)
            actuals = np.concatenate(actuals)
            mask = self.mask_train if split == 'train' else self.mask_val if split == 'val' else self.mask_test
            valid_mask = mask.astype(bool).flatten()

            preds_flat = preds.reshape(-1, 1)
            actuals_flat = actuals.reshape(-1, 1)

            preds_valid = preds_flat[valid_mask.repeat(preds.shape[0], axis=0)]
            actuals_valid = actuals_flat[valid_mask.repeat(actuals.shape[0], axis=0)]

            # Inverse transform
            preds_inv = inverse_transform_predictions(
                pd.DataFrame(preds_valid, columns=[self.cfg['project_setup']['target_variable']]),
                self.cfg['project_setup']['target_variable'],
                self.scaler
            )
            actuals_inv = inverse_transform_predictions(
                pd.DataFrame(actuals_valid, columns=[self.cfg['project_setup']['target_variable']]),
                self.cfg['project_setup']['target_variable'],
                self.scaler
            )
            # Inverse transform
            
            rmse = mean_squared_error(actuals_inv, preds_inv, squared=False)
            mae = mean_absolute_error(actuals_inv, preds_inv)
            r2 = r2_score(actuals_inv, preds_inv)
            metrics[split] = {'rmse': rmse, 'mae': mae, 'r2': r2}
            print(f"{split.capitalize()} Set - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

            
            #save those metrics
        metrics_filename = os.path.join(self.run_output_dir, f'evaluation_metrics.json')
        with open(metrics_filename, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_filename}")
        return metrics
    

    def predict_on_full_data(self):
        from tqdm import tqdm  # make sure this is imported at the top
        self.model.load_state_dict(torch.load(self.final_model_path))
        self.model.eval()

        # Combine datasets
        full_dataset = torch.utils.data.ConcatDataset([
            self.train_dataset,
            self.val_dataset,
            self.test_dataset
        ])
        full_loader = DataLoader(full_dataset, batch_size=8, shuffle=False)

        # Predict
        preds = []
        with torch.no_grad():
            for Xb, _ in full_loader:
                Xb = Xb.float().to(self.device)
                out = self.model(Xb).squeeze().cpu().numpy()
                preds.append(out)

        predicted_grids = np.concatenate(preds)  # [T, H, W]

        # Load full raw df and compute coordinate index
        self.full_df_raw = load_and_prepare_data({
            'data': {
                'raw_data_path': self.raw_path,
                'time_column': self.cfg['data']['time_column']
            }
        })

        # Add row_idx / col_idx
        lat_col = self.cfg['data']['lat_column']
        lon_col = self.cfg['data']['lon_column']
        time_col = self.cfg['data']['time_column']
        fixed_step = self.cfg.get('gridding', {}).get('fixed_step', 0.5)
        lat_min = self.full_df_raw[lat_col].min()
        lon_min = self.full_df_raw[lon_col].min()

        self.full_df_raw['row_idx'] = ((self.full_df_raw[lat_col] - lat_min) / fixed_step).round().astype(int)
        self.full_df_raw['col_idx'] = ((self.full_df_raw[lon_col] - lon_min) / fixed_step).round().astype(int)

        indexed_df = self.full_df_raw.set_index([time_col, 'row_idx', 'col_idx'])

        # Time alignment
        time_steps = np.sort(self.full_df_raw[time_col].unique())
        prediction_times = time_steps[self.n_steps_in:self.n_steps_in + len(predicted_grids)]

        # Coordinate mapping
        coord_cols = ['row_idx', 'col_idx', lat_col, lon_col]
        cell_to_coord = self.full_df_raw[coord_cols].drop_duplicates().set_index(['row_idx', 'col_idx'])

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
        batch_size = trial.suggest_categorical('batch_size', [2, 4, 8, 16])
        n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)
        hidden_channels = trial.suggest_categorical('hidden_channels', [8, 16, 32])
        kernel_options = [1, 3, 5]
        kernel_size = tuple([trial.suggest_categorical(f'kernel_size', kernel_options) for i in range(3)])
        dropout_rate = trial.suggest_float('dropout', 0.0, 0.5)
        use_batchnorm = trial.suggest_categorical('use_batchnorm', [True, False])
        lr = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        n_epochs = self.cfg['cnn3d_params'].get('max_epochs_tune', 10)
        model = CNN3DModel(
            in_channels=len(self.feature_names),
            out_channels=1,
            n_conv_layers=n_conv_layers,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size)
        best_val_loss = float('inf')
        for epoch in range(n_epochs):
            model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.float().to(self.device), yb.float().to(self.device)
                optimizer.zero_grad()
                out = model(Xb).squeeze(1)
                loss = criterion(out, yb.squeeze())
                loss.backward()
                optimizer.step()
            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.float().to(self.device), yb.float().to(self.device)
                    out = model(Xb).squeeze(1)
                    mask = torch.tensor(self.mask_val, device=self.device)  # [H, W]
                    mask = mask.unsqueeze(0).expand(out.shape[0], -1, -1)   # [B, H, W]
                    val_loss = masked_mse(out, yb.squeeze(), mask)

                    val_losses.append(val_loss.item())
            mean_val_loss = np.mean(val_losses)
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
        return best_val_loss

    def tune_and_train(self, n_trials=10):
        study = optuna.create_study(direction='minimize')
        n_trials = self.cfg['optuna_tuning'].get('n_trials', 10)
        study.optimize(self.objective, n_trials=n_trials)
        print('Best trial:', study.best_trial.params)

        # Save best params
        with open(os.path.join(self.run_output_dir, 'best_hyperparams.json'), 'w') as f:
            json.dump(study.best_trial.params, f, indent=4)

        # Use best params
        best_params = study.best_trial.params
        batch_size = best_params['batch_size']
        n_conv_layers = best_params['n_conv_layers']
        hidden_channels = best_params['hidden_channels']
        kernel_size = tuple([best_params['kernel_size']] * 3) if isinstance(best_params['kernel_size'], int) else best_params['kernel_size']
        lr = best_params['learning_rate']
        n_epochs = self.cfg['cnn3d_params'].get('max_epochs', 10)
        patience = self.cfg['cnn3d_params'].get('patience', 10)

        use_batchnorm = self.cfg['cnn3d_params'].get('use_batchnorm', False)
        dropout_rate = self.cfg['cnn3d_params'].get('dropout', 0.0)

        writer = SummaryWriter(log_dir=os.path.join(self.run_output_dir, 'tensorboard_logs_final'))

        model = CNN3DModel(
            in_channels=len(self.feature_names),
            out_channels=1,
            n_conv_layers=n_conv_layers,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            use_batchnorm=use_batchnorm,
            dropout_rate=dropout_rate
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        final_model_path = os.path.join(self.run_models_dir, 'cnn3d_final_model.pt')

        # Save final hyperparams (merged from best trial + config)
        final_hparams = {
            **best_params,
            'max_epochs': n_epochs,
            'patience': patience,
        }
        with open(self.final_model_path.replace('.pt', '_hparams.json'), 'w') as f:
            json.dump(final_hparams, f, indent=4)

        for epoch in range(n_epochs):
            model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.float().to(self.device), yb.float().to(self.device)
                optimizer.zero_grad()
                out = model(Xb).squeeze(1)
                mask = torch.tensor(self.mask_train, device=self.device)
                mask = mask.unsqueeze(0).expand(out.shape[0], -1, -1)  # match batch size
                loss = criterion(out * mask, yb.squeeze() * mask)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.float().to(self.device), yb.float().to(self.device)
                    out = model(Xb).squeeze(1)
                    mask = torch.tensor(self.mask_val, device=self.device)  # [H, W]
                    mask = mask.unsqueeze(0).expand(out.shape[0], -1, -1)   # [B, H, W]
                    val_loss = masked_mse(out, yb.squeeze(), mask)
                    val_losses.append(val_loss.item())
            mean_val_loss = np.mean(val_losses)
            print(f"[Final Training] Epoch {epoch+1}/{n_epochs} - Val Loss: {mean_val_loss:.4f}")
            writer.add_scalar("LossFinal/val", mean_val_loss, epoch)
            writer.add_scalar("LossFinal/train", loss.item(), epoch)

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), final_model_path)
                print(f"✅ Best final model saved to {final_model_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"🛑 Early stopping during final training at epoch {epoch+1}")
                    break

        self.model = model
        writer.close()
        print('Final model training completed.')
#         # Save final model
        torch.save(model.state_dict(), self.final_model_path)
        print(f"Final model saved to {self.final_model_path}")
    def convert_predictions_to_dataframe(self, predicted_grids, target_indices, n_in):
        """
        Convert predicted grids into a DataFrame with time, lat, lon, actuals, predictions, and error metrics.
        Performs inverse scaling using self.target_scalers.
        """
        import tqdm

        time_steps = self.full_df_raw[self.cfg['data']['time_column']].unique()
        pred_start_time_idx = n_in
        prediction_times = time_steps[pred_start_time_idx:pred_start_time_idx + len(predicted_grids)]

        # Ensure coordinate mapping
        if 'row_idx' not in self.full_df_raw.columns or 'col_idx' not in self.full_df_raw.columns:
            grid_cfg = self.cfg.get('gridding', {})
            fixed_step = grid_cfg.get('fixed_step', 0.5)
            lat_min = self.full_df_raw[self.cfg['data']['lat_column']].min()
            lon_min = self.full_df_raw[self.cfg['data']['lon_column']].min()
            self.full_df_raw['row_idx'] = ((self.full_df_raw[self.cfg['data']['lat_column']] - lat_min) / fixed_step).round().astype(int)
            self.full_df_raw['col_idx'] = ((self.full_df_raw[self.cfg['data']['lon_column']] - lon_min) / fixed_step).round().astype(int)

        coord_cols = ['row_idx', 'col_idx', self.cfg['data']['lat_column'], self.cfg['data']['lon_column']]
        cell_to_coord = self.full_df_raw[coord_cols].drop_duplicates().set_index(['row_idx', 'col_idx'])

        valid_pixel_indices = np.argwhere(self.mask == 1)
        output_records = []

        for i, pred_time in enumerate(tqdm.tqdm(prediction_times, desc="Converting predictions with inverse transform")):
            if i >= len(predicted_grids):
                break

            # [C, H, W] or [H, W]
            if predicted_grids.ndim == 5:
                pred_grids = predicted_grids[i, 0]  # [C, H, W]
            elif predicted_grids.ndim == 4:
                pred_grids = predicted_grids[i, 0:1]  # [1, H, W]
            else:
                pred_grids = predicted_grids[i]

            actual_time_idx = pred_start_time_idx + i
            actual_grids = {}
            pred_grids_inv = {}

            for j, target in enumerate(self.target_variables):
                target_idx = target_indices[j]
                # Get actual
                if actual_time_idx < len(self.gridded_data):
                    actual_grid = self.gridded_data[actual_time_idx, :, :, target_idx]
                    if self.target_scalers.get(target):
                        actual_grid = self.target_scalers[target].inverse_transform(actual_grid.reshape(-1, 1)).reshape(actual_grid.shape)
                    actual_grids[target] = actual_grid
                else:
                    actual_grids[target] = None

                # Get prediction
                pred_grid = pred_grids[j] if pred_grids.ndim == 3 else pred_grids
                if self.target_scalers.get(target):
                    pred_grid = self.target_scalers[target].inverse_transform(pred_grid.reshape(-1, 1)).reshape(pred_grid.shape)
                pred_grids_inv[target] = pred_grid

            for r, c in valid_pixel_indices:
                try:
                    coords = cell_to_coord.loc[(r, c)]
                    lat = coords[self.cfg['data']['lat_column']]
                    lon = coords[self.cfg['data']['lon_column']]
                except KeyError:
                    continue

                record = {'time': pred_time, 'lat': lat, 'lon': lon}
                for j, target in enumerate(self.target_variables):
                    pred_val = pred_grids_inv[target][r, c]
                    actual_val = actual_grids[target][r, c] if actual_grids[target] is not None else np.nan
                    record[target] = actual_val
                    record[f'{target}_predicted'] = pred_val
                    record[f'{target}_prediction_error'] = pred_val - actual_val if not np.isnan(actual_val) else np.nan
                    record[f'{target}_absolute_error'] = abs(pred_val - actual_val) if not np.isnan(actual_val) else np.nan
                output_records.append(record)

        return pd.DataFrame(output_records)




# Example usage:
# pipeline = CNN3DPipeline('config/cnn3d/config_CNN3D_SPEI.yaml')
# pipeline.train()
# metrics = pipeline.evaluate()
# pipeline.predict_on_full_data()