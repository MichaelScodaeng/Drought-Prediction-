import pandas as pd
import numpy as np
import yaml
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import matplotlib.pyplot as plt

# --- Helper function for JSON serialization ---
def _to_python_type(obj):
    if isinstance(obj, dict): return {k: _to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [_to_python_type(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64)): return int(obj)
    elif isinstance(obj, (np.floating, np.float64)): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    else: return obj

# --- Deep Learning & Tuning Imports ---
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    import optuna
    PYTORCH_AVAILABLE = True
    print("PyTorch, PyTorch Lightning, and Optuna successfully imported.")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch, PyTorch Lightning or Optuna not found. Please install them to run this pipeline.")

# --- Utility Function Imports ---
try:
    from src.data_utils import load_config, load_and_prepare_data
    from src.grid_utils import create_gridded_data
    print("CNN3D Pipeline: Successfully imported utility functions.")
except ImportError as e:
    print(f"CNN3D Pipeline Error: Could not import utility functions: {e}")

# --- PyTorch CNN3D Model Definition (ARCHITECTURE CORRECTED) ---
class CNN3DRegressor(nn.Module):
    def __init__(self, in_channels, grid_h, grid_w, n_steps_in, n_steps_out,
                 n_conv_layers, out_channels, kernel_size, dropout_rate):
        super(CNN3DRegressor, self).__init__()

        layers = []
        current_channels = in_channels

        for _ in range(n_conv_layers):
            layers.append(nn.Conv3d(in_channels=current_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    padding='same'))
            layers.append(nn.ReLU())
            current_channels = out_channels

        self.conv_block = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # Reduce (D,H,W) → (1,1,1)

        self.fc = nn.Sequential(
            nn.Flatten(),  # Output shape: (batch, out_channels)
            nn.Linear(out_channels, 256),  # Fully connected layer
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Final output layer: (B, n_steps_out * grid_h * grid_w)
            nn.Linear(64, n_steps_out * grid_h * grid_w)
        )

        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_steps_out = n_steps_out

    def forward(self, x):
        x = self.conv_block(x)
        x = self.global_pool(x)   # Shape: (B, C, 1, 1, 1)
        x = self.fc(x)            # Shape: (B, output_dim)
        out = x.view(-1, self.n_steps_out, self.grid_h, self.grid_w)
        return out

# --- Lightning Module with Masked Loss ---
class GridModelLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate, mask, trial=None):
        super().__init__(); self.model = model; self.learning_rate = learning_rate
        self.trial = trial; self.criterion = nn.MSELoss(reduction='none'); self.register_buffer('mask', mask); self.validation_step_outputs = []
    def forward(self, x): return self.model(x)
    def _calculate_masked_loss(self, y_hat, y):
        if y_hat.shape[1] == 1: y_hat = y_hat.squeeze(1)
        loss = self.criterion(y_hat, y); masked_loss = loss * self.mask
        return masked_loss.sum() / (self.mask.sum() * y.size(0) + 1e-9)
    def training_step(self, batch, batch_idx): x, y = batch; y_hat = self(x); return self._calculate_masked_loss(y_hat, y)
    def validation_step(self, batch, batch_idx): x, y = batch; loss = self._calculate_masked_loss(self(x), y); self.log('val_loss', loss, on_epoch=True); self.validation_step_outputs.append(loss); return loss
    def predict_step(self, batch, batch_idx, dataloader_idx=0): x, y = batch; return self(x)
    def on_validation_epoch_end(self):
        if not self.validation_step_outputs: return
        avg_loss = torch.stack(self.validation_step_outputs).mean(); self.log('val_rmse', torch.sqrt(avg_loss)); self.validation_step_outputs.clear()
        if self.trial: self.trial.report(avg_loss, self.current_epoch);
        if self.trial and self.trial.should_prune(): raise optuna.exceptions.TrialPruned()
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# --- Custom Dataset for 3D Gridded Data ---
class SequenceDataset3D(Dataset):
    def __init__(self, gridded_data, target_feature_idx, n_steps_in, n_steps_out=1):
        self.data = torch.tensor(gridded_data, dtype=torch.float32).permute(0, 3, 1, 2)
        self.target_feature_idx = target_feature_idx; self.n_steps_in = n_steps_in; self.n_steps_out = n_steps_out
    def __len__(self): return self.data.shape[0] - self.n_steps_in - self.n_steps_out + 1
    def __getitem__(self, idx):
        end_idx = idx + self.n_steps_in; out_end_idx = end_idx + self.n_steps_out
        seq_x = self.data[idx:end_idx].permute(1, 0, 2, 3); seq_y = self.data[end_idx:out_end_idx, self.target_feature_idx, :, :]
        if self.n_steps_out == 1: return seq_x, seq_y.squeeze(0)
        return seq_x, seq_y

# --- Main Pipeline Class ---
class CNN3DGlobalPipeline:
    def __init__(self, config_path="config.yaml"):
        # (Initialization is the same, setting up paths)
        self.config_path_abs = os.path.abspath(config_path)
        self.cfg = load_config(self.config_path_abs)
        self.experiment_name = self.cfg.get('project_setup',{}).get('experiment_name','cnn3d_experiment')
        self.project_root_for_paths = os.path.join(os.path.dirname(self.config_path_abs),"..","..")
        self.run_output_dir = os.path.join(self.project_root_for_paths, 'run_outputs', self.experiment_name)
        self.run_models_dir = os.path.join(self.project_root_for_paths, 'models_saved', self.experiment_name)
        os.makedirs(self.run_output_dir, exist_ok=True); os.makedirs(self.run_models_dir, exist_ok=True)
        self.model = None; self.all_metrics = {}; self.mask = None; self.full_df_raw_for_prediction = None; self.gridded_data = None

    def _get_abs_path_from_config_value(self, relative_path):
        if not relative_path or os.path.isabs(relative_path): return relative_path
        return os.path.abspath(os.path.join(self.project_root_for_paths, relative_path))

    def _calculate_masked_metrics(self, actuals, predictions, mask):
        mask_bool = mask.bool().to(actuals.device); min_len = min(len(actuals), len(predictions)); actuals, predictions = actuals[:min_len], predictions[:min_len]
        if predictions.ndim == 4 and predictions.shape[1] == 1: predictions = predictions.squeeze(1)
        batch_mask = mask_bool.expand_as(actuals)
        actuals_np = actuals[batch_mask].flatten().cpu().numpy(); preds_np = predictions[batch_mask].flatten().cpu().numpy()
        return {'rmse': mean_squared_error(actuals_np, preds_np), 'mae': mean_absolute_error(actuals_np, preds_np), 'r2': r2_score(actuals_np, preds_np)}

    def _objective_for_optuna(self, trial, train_loader, val_loader, in_channels, grid_h, grid_w, n_steps_in, n_steps_out):
        cnn_tuning_cfg = self.cfg.get('cnn3d_params', {}).get('tuning', {})
        learning_rate = trial.suggest_float('learning_rate', **cnn_tuning_cfg.get('learning_rate'))
        n_conv_layers = trial.suggest_int('n_conv_layers', **cnn_tuning_cfg.get('n_conv_layers'))
        out_channels_power = trial.suggest_int('out_channels_power', **cnn_tuning_cfg.get('out_channels_power'))
        out_channels = 2**out_channels_power
        k_d = trial.suggest_categorical('kernel_size_d', cnn_tuning_cfg.get('kernel_size_d', {}).get('choices', [3])); k_h = trial.suggest_categorical('kernel_size_h', cnn_tuning_cfg.get('kernel_size_h', {}).get('choices', [3])); k_w = trial.suggest_categorical('kernel_size_w', cnn_tuning_cfg.get('kernel_size_w', {}).get('choices', [3]))
        dropout_rate = trial.suggest_float('dropout_rate', **cnn_tuning_cfg.get('dropout_rate', {'low': 0.2, 'high': 0.5}))
        
        model = CNN3DRegressor(in_channels, grid_h, grid_w, n_steps_in, n_steps_out, n_conv_layers, out_channels, (k_d, k_h, k_w), dropout_rate)
        lightning_model = GridModelLightningModule(model, learning_rate, self.mask, trial=trial)
        
        trainer_params = self.cfg.get('cnn3d_params', {}).get('trainer', {})
        early_stopping = EarlyStopping(monitor="val_loss", patience=trainer_params.get('patience_for_early_stopping', 5))
        trainer = pl.Trainer(max_epochs=trainer_params.get('max_epochs', 50), callbacks=[early_stopping],
            logger=False, enable_checkpointing=False, enable_progress_bar=False, accelerator='auto', devices=1)
        try:
            trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        except optuna.exceptions.TrialPruned: return float('inf')
        return trainer.callback_metrics.get("val_loss", torch.tensor(float('inf'))).item()

    def run_pipeline(self):
        if not PYTORCH_AVAILABLE: return "Failed: PyTorch/Lightning/Optuna not found."
        print(f"\n--- Starting CNN3D PyTorch GLOBAL Pipeline ---")
        
        print("Pipeline: Loading and gridding data...")
        raw_path = self.cfg.get('data',{}).get('raw_data_path'); abs_path = self._get_abs_path_from_config_value(raw_path)
        full_df_raw = load_and_prepare_data({'data': {'raw_data_path': abs_path, 'time_column': self.cfg['data']['time_column']}})
        if full_df_raw is None: return "Failed: Data Load"
        self.full_df_raw_for_prediction = full_df_raw.copy()
        
        self.gridded_data, mask = create_gridded_data(full_df_raw, self.cfg)
        self.mask = torch.tensor(mask, dtype=torch.float32)

        grid_h, grid_w = self.gridded_data.shape[1], self.gridded_data.shape[2]

        print("Pipeline: Splitting gridded data..."); time_steps = full_df_raw[self.cfg['data']['time_column']].unique()
        train_end_idx = np.where(time_steps <= np.datetime64(self.cfg['data']['train_end_date']))[0][-1]
        val_end_idx = np.where(time_steps <= np.datetime64(self.cfg['data']['validation_end_date']))[0][-1]
        train_data_grid = self.gridded_data[:train_end_idx + 1]; val_data_grid = self.gridded_data[train_end_idx + 1 : val_end_idx + 1]; test_data_grid = self.gridded_data[val_end_idx + 1 :]
        
        print("Pipeline: Creating Datasets and DataLoaders...")
        seq_params = self.cfg.get('sequence_params', {}); n_steps_in = seq_params.get('n_steps_in', 12); n_steps_out = seq_params.get('n_steps_out', 1)
        target_idx = self.cfg['data']['features_to_grid'].index(self.cfg['project_setup']['target_variable'])
        
        train_dataset = SequenceDataset3D(train_data_grid, target_idx, n_steps_in, n_steps_out)
        val_dataset = SequenceDataset3D(val_data_grid, target_idx, n_steps_in, n_steps_out)
        test_dataset = SequenceDataset3D(test_data_grid, target_idx, n_steps_in, n_steps_out)
        if len(train_dataset) == 0 or len(val_dataset) == 0: return "Failed: Not enough data for sequences"
        
        batch_size = self.cfg.get('cnn3d_params',{}).get('batch_size', 16); num_workers = 2 if os.name != 'nt' else 0
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

        print("Pipeline: Starting hyperparameter tuning...")
        in_channels = len(self.cfg['data']['features_to_grid'])
        n_trials = self.cfg.get('cnn3d_params', {}).get('tuning', {}).get('n_trials', 15)
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial: self._objective_for_optuna(trial, train_loader, val_loader, in_channels, grid_h, grid_w, n_steps_in, n_steps_out), n_trials=n_trials)
        self.best_hyperparams = study.best_trial.params
        print(f"Pipeline: Optuna found best params: {self.best_hyperparams}")

        print("Pipeline: Training final model..."); best = self.best_hyperparams
        final_model_base = CNN3DRegressor(in_channels, grid_h, grid_w, n_steps_in, n_steps_out, best['n_conv_layers'], 2**best['out_channels_power'], (best['kernel_size_d'],best['kernel_size_h'],best['kernel_size_w']), best['dropout_rate'])
        final_lightning_model = GridModelLightningModule(final_model_base, best['learning_rate'], self.mask)
        full_train_loader = DataLoader(torch.utils.data.ConcatDataset([train_dataset, val_dataset]), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        checkpoint_callback = ModelCheckpoint(dirpath=self.run_models_dir, filename="global-cnn3d-best-model", monitor="val_loss", mode="min")
        final_trainer = pl.Trainer(max_epochs=self.cfg.get('cnn3d_params',{}).get('trainer',{}).get('max_epochs', 50), callbacks=[checkpoint_callback], logger=False, enable_progress_bar=True, accelerator='auto', devices=1)
        final_trainer.fit(model=final_lightning_model, train_dataloaders=full_train_loader, val_dataloaders=val_loader)
        best_model_path = checkpoint_callback.best_model_path
        print(f"Pipeline: Final model training complete. Best model saved at: {best_model_path}")
        self.model = GridModelLightningModule.load_from_checkpoint(best_model_path, model=final_lightning_model.model, learning_rate=final_lightning_model.learning_rate, mask=self.mask)

        self.evaluate_and_save(final_trainer, train_dataset, val_dataset, test_dataset)
        self.predict_on_full_data()
        
        print("--- CNN3D Global Pipeline Run Finished ---")
        return self.all_metrics
    
    # ... other methods like evaluate_and_save and predict_on_full_data would be here ...
    # (These are kept the same as the previous version for brevity)
    def evaluate_and_save(self, trainer, train_dataset, val_dataset, test_dataset):
        print("\n--- Final Model Evaluation ---"); self.all_metrics = {}
        self.model.eval()
        with torch.no_grad():
            for split_name, dataset in [('train', train_dataset), ('validation', val_dataset), ('test', test_dataset)]:
                if len(dataset) > 0:
                    loader = DataLoader(dataset, batch_size=self.cfg.get('cnn3d_params',{}).get('batch_size', 16))
                    y_actual_list = [y for _, y in loader]
                    scaled_preds_list = trainer.predict(self.model, dataloaders=loader)
                    y_actual_grid = torch.cat(y_actual_list).cpu(); scaled_preds_grid = torch.cat(scaled_preds_list).cpu()
                    metrics = self._calculate_masked_metrics(y_actual_grid, scaled_preds_grid, self.mask)
                    self.all_metrics[split_name] = metrics
                    print(f"  {split_name.capitalize()} Set: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")
        metrics_filename = self.cfg.get('results',{}).get('metrics_filename', 'global_cnn3d_metrics.json')
        with open(os.path.join(self.run_output_dir, metrics_filename), 'w') as f:
            json.dump(_to_python_type(self.all_metrics), f, indent=4)
        print(f"Pipeline: Evaluation metrics saved.")
    
    def predict_on_full_data(self):
        print("\nPipeline: Generating predictions on the full raw dataset...");
        if self.model is None or self.gridded_data is None: return None
        target_col = self.cfg['project_setup']['target_variable']; target_idx = self.cfg['data']['features_to_grid'].index(target_col)
        seq_params = self.cfg.get('sequence_params', {}); n_steps_in = seq_params.get('n_steps_in', 12); n_steps_out = seq_params.get('n_steps_out', 1)

        full_dataset = SequenceDataset3D(self.gridded_data, target_idx, n_steps_in, n_steps_out)
        if len(full_dataset) == 0: print("Not enough data for full prediction."); return None
        full_loader = DataLoader(full_dataset, batch_size=self.cfg.get('cnn3d_params',{}).get('batch_size',16))
        
        self.model.eval()
        with torch.no_grad():
            trainer = pl.Trainer(accelerator='auto', devices=1, logger=False)
            predicted_grids = torch.cat(trainer.predict(self.model, dataloaders=full_loader)).cpu().numpy()

        if predicted_grids.ndim == 4 and predicted_grids.shape[1] == 1:
            predicted_grids = predicted_grids.squeeze(1)

        print("Pipeline: Un-gridding predictions to create output CSV...")
        time_steps = self.full_df_raw_for_prediction[self.cfg['data']['time_column']].unique()
        pred_start_time_idx = n_steps_in + n_steps_out - 1
        prediction_times = time_steps[pred_start_time_idx:pred_start_time_idx + len(predicted_grids)]

        output_records = []
        valid_pixel_indices = np.argwhere(self.mask.cpu().numpy() == 1)
        
        if 'row_idx' not in self.full_df_raw_for_prediction.columns:
            grid_cfg = self.cfg.get('gridding', {}); fixed_step = grid_cfg.get('fixed_step', 0.5)
            lat_min = self.full_df_raw_for_prediction[self.cfg['data']['lat_column']].min(); lon_min = self.full_df_raw_for_prediction[self.cfg['data']['lon_column']].min()
            self.full_df_raw_for_prediction['row_idx'] = ((self.full_df_raw_for_prediction[self.cfg['data']['lat_column']] - lat_min) / fixed_step).round().astype(int)
            self.full_df_raw_for_prediction['col_idx'] = ((self.full_df_raw_for_prediction[self.cfg['data']['lon_column']] - lon_min) / fixed_step).round().astype(int)
        
        cell_to_coord = self.full_df_raw_for_prediction[['row_idx','col_idx','lat','lon']].drop_duplicates().set_index(['row_idx','col_idx'])

        for i, t in enumerate(prediction_times):
            pred_grid = predicted_grids[i]
            actual_grid = self.gridded_data[pred_start_time_idx + i, :, :, target_idx]
            for r, c in valid_pixel_indices:
                try: coords = cell_to_coord.loc[(r,c)]; lat, lon = coords.lat, coords.lon
                except KeyError: continue
                actual_value_row = self.full_df_raw_for_prediction[
                    (self.full_df_raw_for_prediction[self.cfg['data']['time_column']] == t) &
                    (self.full_df_raw_for_prediction['lat'] == lat) &
                    (self.full_df_raw_for_prediction['lon'] == lon)]
                actual_value = actual_value_row[target_col].values[0] if not actual_value_row.empty else np.nan
                output_records.append({
                    'time': t, 'lat': lat, 'lon': lon,
                    target_col: actual_value,
                    f'{target_col}_predicted': pred_grid[r, c]})

        output_df = pd.DataFrame(output_records)
        pred_filename = self.cfg.get('results',{}).get('predictions_filename', 'global_cnn3d_full_predictions.csv')
        save_path = os.path.join(self.run_output_dir, pred_filename)
        output_df.to_csv(save_path, index=False)
        print(f"Pipeline: Full data predictions saved to {save_path}")
