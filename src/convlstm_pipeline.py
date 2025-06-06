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
    print("ConvLSTM Pipeline: Successfully imported utility functions.")
except ImportError as e:
    print(f"ConvLSTM Pipeline Error: Could not import utility functions: {e}")

# --- PyTorch ConvLSTM Model Definition (CORRECTED) ---
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim, out_channels=4 * hidden_dim,
                              kernel_size=kernel_size, padding=padding, bias=bias)
    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i); f = torch.sigmoid(cc_f); o = torch.sigmoid(cc_o); g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g; h_next = o * torch.tanh(c_next)
        return h_next, c_next
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers; self.batch_first = batch_first; cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x, hidden_state=None):
        # Input x shape: (N, T, C, H, W) if batch_first=True
        if self.batch_first:
            x = x.permute(1, 0, 2, 3, 4) # now x has shape (T, N, C, H, W)
        
        # --- FIX START ---
        # Correctly unpack the shape
        seq_len, batch_size, _, h, w = x.size()
        # --- FIX END ---
        
        if hidden_state is None:
            # Initialize hidden state with the correct batch_size
            hidden_state = [cell.init_hidden(batch_size, (h, w)) for cell in self.cell_list]

        layer_output_list, last_state_list = [], []
        cur_layer_input = x
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](x=cur_layer_input[t, :, :, :, :], cur_state=[h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=0); cur_layer_input = layer_output
            layer_output_list.append(layer_output); last_state_list.append([h, c])
        
        # We are interested in the output of the last layer
        final_layer_output = layer_output_list[-1]
        
        if self.batch_first:
            final_layer_output = final_layer_output.permute(1, 0, 2, 3, 4) # (N, T, C_hidden, H, W)
            
        return final_layer_output, last_state_list

class ConvLSTMRegressor(nn.Module):
    def __init__(self, in_channels, n_steps_out, n_layers, hidden_channels, kernel_size):
        super(ConvLSTMRegressor, self).__init__()
        self.convlstm = ConvLSTM(input_dim=in_channels, hidden_dim=hidden_channels, kernel_size=kernel_size, num_layers=n_layers, batch_first=True)
        self.final_conv = nn.Conv2d(in_channels=hidden_channels, out_channels=n_steps_out, kernel_size=1)
    def forward(self, x):
        layer_output, _ = self.convlstm(x)
        last_time_step_output = layer_output[:, -1, :, :, :]
        out = self.final_conv(last_time_step_output)
        return out

# --- Lightning Module with Masked Loss ---
class GridModelLightningModule(pl.LightningModule):
    # This class can be reused from the CNN3D pipeline without changes
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

# --- Custom Dataset for ConvLSTM (needs different permutation) ---
class SequenceDatasetConvLSTM(Dataset):
    def __init__(self, gridded_data, target_feature_idx, n_steps_in, n_steps_out=1):
        # ConvLSTM expects (T, C, H, W) -> we permute to this from (T, H, W, C)
        self.data = torch.tensor(gridded_data, dtype=torch.float32).permute(0, 3, 1, 2)
        self.target_feature_idx = target_feature_idx; self.n_steps_in = n_steps_in; self.n_steps_out = n_steps_out
    def __len__(self): return self.data.shape[0] - self.n_steps_in - self.n_steps_out + 1
    def __getitem__(self, idx):
        end_idx = idx + self.n_steps_in; out_end_idx = end_idx + self.n_steps_out
        # Input seq_x shape: (T, C, H, W)
        seq_x = self.data[idx:end_idx]
        seq_y = self.data[end_idx:out_end_idx, self.target_feature_idx, :, :]
        if self.n_steps_out == 1: return seq_x, seq_y.squeeze(0)
        return seq_x, seq_y

# --- Main Pipeline Class ---
class ConvLSTMPipeline:
    # This class structure is very similar to the CNN3D one, with adapted model and config keys
    def __init__(self, config_path="config.yaml"):
        # (This remains the same)
        self.config_path_abs = os.path.abspath(config_path); self.cfg = load_config(self.config_path_abs)
        self.experiment_name = self.cfg.get('project_setup',{}).get('experiment_name','convlstm_experiment')
        self.project_root_for_paths = os.path.dirname(self.config_path_abs)
        self.run_output_dir = os.path.join(self.project_root_for_paths, 'run_outputs', self.experiment_name)
        self.run_models_dir = os.path.join(self.project_root_for_paths, 'models_saved', self.experiment_name)
        os.makedirs(self.run_output_dir, exist_ok=True); os.makedirs(self.run_models_dir, exist_ok=True)
        per_loc_preds_dir_name = self.cfg.get('results', {}).get('per_location_predictions_dir', 'per_location_full_predictions')
        self.per_location_predictions_output_dir = os.path.join(self.run_output_dir, per_loc_preds_dir_name)
        os.makedirs(self.per_location_predictions_output_dir, exist_ok=True)
        self.model = None; self.all_metrics = {}; self.mask = None; self.full_df_raw = None; self.gridded_data = None

    def _get_abs_path_from_config_value(self, rp): return os.path.abspath(os.path.join(self.project_root_for_paths, rp)) if rp and not os.path.isabs(rp) else rp
    def _calculate_masked_metrics(self, actuals, preds, mask):
        # (This remains the same)
        mask_bool = mask.bool().to(actuals.device); min_len = min(len(actuals), len(preds)); actuals, preds = actuals[:min_len], preds[:min_len]
        if preds.ndim == 4 and preds.shape[1] == 1: preds = preds.squeeze(1)
        batch_mask = mask_bool.expand_as(actuals)
        actuals_np = actuals[batch_mask].flatten().cpu().numpy(); preds_np = preds[batch_mask].flatten().cpu().numpy()
        return {'rmse': mean_squared_error(actuals_np, preds_np, squared=False), 'mae': mean_absolute_error(actuals_np, preds_np), 'r2': r2_score(actuals_np, preds_np)}

    def _objective(self, trial, train_loader, val_loader, in_channels, n_steps_out):
        # (This remains the same, but with convlstm_params)
        cfg = self.cfg.get('convlstm_params', {}).get('tuning', {})
        lr = trial.suggest_float('learning_rate', **cfg.get('learning_rate'))
        n_layers = trial.suggest_int('n_layers', **cfg.get('n_layers'))
        hidden_channels = trial.suggest_categorical('hidden_channels', cfg.get('hidden_channels',{}).get('choices', [16,32]))
        kernel_size = trial.suggest_categorical('kernel_size', cfg.get('kernel_size',{}).get('choices', [3,5]))
        model = ConvLSTMRegressor(in_channels, n_steps_out, n_layers, hidden_channels, kernel_size)
        lightning_model = GridModelLightningModule(model, lr, self.mask, trial=trial)
        trainer_params = self.cfg.get('convlstm_params', {}).get('trainer', {})
        early_stopping = EarlyStopping(monitor="val_loss", patience=trainer_params.get('patience_for_early_stopping', 5))
        trainer = pl.Trainer(max_epochs=trainer_params.get('max_epochs', 50), callbacks=[early_stopping], logger=False, enable_checkpointing=False, enable_progress_bar=False, accelerator='auto', devices=1)
        try: trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        except optuna.exceptions.TrialPruned: return float('inf')
        return trainer.callback_metrics.get("val_loss", torch.tensor(float('inf'))).item()

    def run_pipeline(self):
        # (This remains the same, just points to convlstm_params in config)
        if not PYTORCH_AVAILABLE: return "Failed: Dependencies not found."
        print(f"\n--- Starting ConvLSTM Global Pipeline ---");
        raw_path = self._get_abs_path_from_config_value(self.cfg.get('data',{}).get('raw_data_path'))
        self.full_df_raw = load_and_prepare_data({'data': {'raw_data_path': raw_path, 'time_column': self.cfg['data']['time_column']}})
        if self.full_df_raw is None: return "Failed: Data Load"
        
        self.gridded_data, mask = create_gridded_data(self.full_df_raw, self.cfg)
        self.mask = torch.tensor(mask, dtype=torch.float32)

        time_steps = self.full_df_raw[self.cfg['data']['time_column']].unique()
        train_end_idx = np.where(time_steps <= np.datetime64(self.cfg['data']['train_end_date']))[0][-1]
        val_end_idx = np.where(time_steps <= np.datetime64(self.cfg['data']['validation_end_date']))[0][-1]
        train_grid, val_grid, test_grid = self.gridded_data[:train_end_idx+1], self.gridded_data[train_end_idx+1:val_end_idx+1], self.gridded_data[val_end_idx+1:]
        
        seq_cfg = self.cfg.get('sequence_params', {}); n_in, n_out = seq_cfg.get('n_steps_in',12), seq_cfg.get('n_steps_out',1)
        target_idx = self.cfg['data']['features_to_grid'].index(self.cfg['project_setup']['target_variable'])
        
        train_ds = SequenceDatasetConvLSTM(train_grid, target_idx, n_in, n_out)
        val_ds = SequenceDatasetConvLSTM(val_grid, target_idx, n_in, n_out)
        test_ds = SequenceDatasetConvLSTM(test_grid, target_idx, n_in, n_out)
        num_workers = 8 if os.name != 'nt' else 0
        
        if len(train_ds) == 0 or len(val_ds) == 0: return "Failed: Not enough data for sequences"
        
        batch_size = self.cfg.get('convlstm_params',{}).get('batch_size',8); num_workers = num_workers if os.name != 'nt' else 0
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda t: self._objective(t, train_loader, val_loader, len(self.cfg['data']['features_to_grid']), n_out), 
                       n_trials=self.cfg.get('convlstm_params',{}).get('tuning',{}).get('n_trials', 15))
        self.best_hyperparams = study.best_trial.params
        print(f"Pipeline: Optuna found best params: {self.best_hyperparams}")
        
        best = self.best_hyperparams
        final_model_base = ConvLSTMRegressor(len(self.cfg['data']['features_to_grid']), n_out, best['n_layers'], best['hidden_channels'], best['kernel_size'])
        final_lightning_model = GridModelLightningModule(final_model_base, best['learning_rate'], self.mask)
        full_train_loader = DataLoader(torch.utils.data.ConcatDataset([train_ds, val_ds]), batch_size=batch_size, shuffle=True)
        
        ckpt_cb = ModelCheckpoint(dirpath=self.run_models_dir, filename="global-convlstm-best", monitor="val_loss", mode="min")
        trainer_cfg = self.cfg.get('convlstm_params',{}).get('trainer',{})
        final_trainer = pl.Trainer(max_epochs=trainer_cfg.get('max_epochs',50), callbacks=[ckpt_cb], logger=False, 
                                   enable_progress_bar=trainer_cfg.get('enable_progress_bar',True), accelerator='auto', devices=1)
        final_trainer.fit(model=final_lightning_model, train_dataloaders=full_train_loader, val_dataloaders=val_loader)
        
        best_model_path = ckpt_cb.best_model_path
        print(f"Pipeline: Final model training complete. Best model saved at: {best_model_path}")
        self.model = GridModelLightningModule.load_from_checkpoint(best_model_path, model=final_lightning_model.model, learning_rate=final_lightning_model.learning_rate, mask=self.mask)

        self.evaluate_and_save(final_trainer, train_ds, val_ds, test_ds)
        self.predict_on_full_data()
        
        print(f"--- ConvLSTM Global Pipeline Run Finished ---")
        return self.all_metrics

    # ... The evaluate_and_save and predict_on_full_data methods are added here, complete and working ...
    def evaluate_and_save(self, trainer, train_dataset, val_dataset, test_dataset):
        print("\n--- Final Model Evaluation ---")
        self.all_metrics = {}
        self.model.eval()
        with torch.no_grad():
            for split_name, dataset in [('train', train_dataset), ('validation', val_dataset), ('test', test_dataset)]:
                if len(dataset) > 0:
                    loader = DataLoader(dataset, batch_size=self.cfg.get('convlstm_params',{}).get('batch_size', 8))
                    y_actual_list, scaled_preds_list = [], []
                    for _, y_batch in loader: y_actual_list.append(y_batch)
                    scaled_preds_list = trainer.predict(self.model, dataloaders=loader)
                    y_actual_grid = torch.cat(y_actual_list).cpu(); scaled_preds_grid = torch.cat(scaled_preds_list).cpu()
                    metrics = self._calculate_masked_metrics(y_actual_grid, scaled_preds_grid, self.mask)
                    self.all_metrics[split_name] = metrics
                    print(f"  {split_name.capitalize()} Set: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")
        
        metrics_filename = self.cfg.get('results',{}).get('metrics_filename', 'global_convlstm_metrics.json')
        with open(os.path.join(self.run_output_dir, metrics_filename), 'w') as f:
            json.dump(_to_python_type(self.all_metrics), f, indent=4)
        print(f"Pipeline: Evaluation metrics saved.")
    
    def predict_on_full_data(self):
        print("\nPipeline: Generating predictions on the full raw dataset...")
        if self.model is None or self.gridded_data is None: return None
        target_col = self.cfg['project_setup']['target_variable']; target_idx = self.cfg['data']['features_to_grid'].index(target_col)
        seq_params = self.cfg.get('sequence_params', {}); n_steps_in = seq_params.get('n_steps_in', 12); n_steps_out = seq_params.get('n_steps_out', 1)
        full_dataset = SequenceDatasetConvLSTM(self.gridded_data, target_idx, n_steps_in, n_steps_out)
        if len(full_dataset) == 0: print("Not enough data for full prediction."); return None
        full_loader = DataLoader(full_dataset, batch_size=self.cfg.get('convlstm_params',{}).get('batch_size',8))
        self.model.eval()
        with torch.no_grad():
            trainer = pl.Trainer(accelerator='auto', devices=1, logger=False)
            predicted_grids = torch.cat(trainer.predict(self.model, dataloaders=full_loader)).cpu().numpy()
        if predicted_grids.ndim == 4 and predicted_grids.shape[1] == 1: predicted_grids = predicted_grids.squeeze(1)
        print("Pipeline: Un-gridding predictions to create output CSV...")
        time_steps = self.full_df_raw[self.cfg['data']['time_column']].unique()
        pred_start_time_idx = n_steps_in + n_steps_out - 1
        prediction_times = time_steps[pred_start_time_idx:pred_start_time_idx + len(predicted_grids)]
        output_records = []
        valid_pixel_indices = np.argwhere(self.mask.cpu().numpy() == 1)
        if 'row_idx' not in self.full_df_raw.columns:
            grid_cfg = self.cfg.get('gridding', {}); fixed_step = grid_cfg.get('fixed_step', 0.5)
            lat_min = self.full_df_raw[self.cfg['data']['lat_column']].min(); lon_min = self.full_df_raw[self.cfg['data']['lon_column']].min()
            self.full_df_raw['row_idx'] = ((self.full_df_raw[self.cfg['data']['lat_column']] - lat_min) / fixed_step).round().astype(int)
            self.full_df_raw['col_idx'] = ((self.full_df_raw[self.cfg['data']['lon_column']] - lon_min) / fixed_step).round().astype(int)
        cell_to_coord = self.full_df_raw[['row_idx','col_idx','lat','lon']].drop_duplicates().set_index(['row_idx','col_idx'])
        for i, t in enumerate(prediction_times):
            pred_grid = predicted_grids[i]
            actual_grid = self.gridded_data[pred_start_time_idx + i, :, :, target_idx]
            for r, c in valid_pixel_indices:
                try: coords = cell_to_coord.loc[(r,c)]; lat, lon = coords.lat, coords.lon
                except KeyError: continue
                actual_value_row = self.full_df_raw[(self.full_df_raw[self.cfg['data']['time_column']] == t) & (self.full_df_raw['lat'] == lat) & (self.full_df_raw['lon'] == lon)]
                actual_value = actual_value_row[target_col].values[0] if not actual_value_row.empty else np.nan
                output_records.append({'time': t, 'lat': lat, 'lon': lon, target_col: actual_value, f'{target_col}_predicted': pred_grid[r, c]})
        output_df = pd.DataFrame(output_records)
        pred_filename = self.cfg.get('results',{}).get('predictions_filename', 'global_convlstm_full_predictions.csv')
        save_path = os.path.join(self.run_output_dir, pred_filename)
        output_df.to_csv(save_path, index=False)
        print(f"Pipeline: Full data predictions saved to {save_path}")
