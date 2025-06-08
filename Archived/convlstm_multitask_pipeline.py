import pandas as pd
import numpy as np
import yaml
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from tqdm import tqdm # Import tqdm for the progress bar

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
    print("ConvLSTM Multitask Pipeline: Successfully imported utility functions.")
except ImportError as e:
    print(f"ConvLSTM Multitask Pipeline Error: Could not import utility functions: {e}")

# --- User-Provided ConvLSTM Model Architecture ---
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim; self.kernel_size = kernel_size; self.bias = bias
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim, out_channels=4 * hidden_dim, kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state; combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined); cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i); f = torch.sigmoid(cc_f); o = torch.sigmoid(cc_o); g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g; h_next = o * torch.tanh(c_next)
        # Initialize layer norm if not done yet
        if self.layer_norm is None:
            _, _, H, W = h_next.shape
            self.layer_norm = nn.LayerNorm([self.hidden_dim, H, W], elementwise_affine=True).to(h_next.device)

        h_next = self.layer_norm(h_next)
        return h_next, c_next
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device), torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True, return_all_layers=True):
        super(ConvLSTM, self).__init__()
        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers); hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers: raise ValueError('Inconsistent list length.')
        self.input_dim = input_dim; self.hidden_dim = hidden_dim; self.kernel_size = kernel_size; self.num_layers = num_layers
        self.batch_first = batch_first; self.bias = bias; self.return_all_layers = return_all_layers
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim, hidden_dim=self.hidden_dim[i], kernel_size=self.kernel_size[i], bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)
    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first: input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()
        if hidden_state is None: hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        layer_output_list = []; last_state_list = []
        seq_len = input_tensor.size(1); cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]; output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1); cur_layer_input = layer_output
            layer_output_list.append(layer_output); last_state_list.append([h, c])
        if not self.return_all_layers: layer_output_list = layer_output_list[-1:]; last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list
    def _init_hidden(self, batch_size, image_size):
        init_states = [];
        for i in range(self.num_layers): init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states
    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list): param = [param] * num_layers
        return param

class EncodingForecastingConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, pre_seq_length, aft_seq_length, n_targets, batch_first=True):
        super().__init__(); self.hidden_dim = hidden_dim; self.pre_seq_length = pre_seq_length; self.aft_seq_length = aft_seq_length
        self.encoder = ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers, batch_first=batch_first, return_all_layers=True)
        self.forecaster = ConvLSTM(input_dim=hidden_dim[-1] if isinstance(hidden_dim, list) else hidden_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers, batch_first=batch_first, return_all_layers=False)
        self.conv_last = nn.Conv2d(in_channels=hidden_dim[-1] if isinstance(hidden_dim, list) else hidden_dim, out_channels=n_targets, kernel_size=1)
    def forward(self, input_tensor):
        _, encoder_states = self.encoder(input_tensor); forecaster_states = encoder_states
        next_input = encoder_states[-1][0].unsqueeze(1)
        predictions = []
        for _ in range(self.aft_seq_length):
            forecaster_output, forecaster_states = self.forecaster(next_input, forecaster_states)
            hidden_state = forecaster_output[0][:, -1]
            
            # Residual connection: add input from previous step if dimensions match
            if hidden_state.shape == next_input.squeeze(1).shape:
                hidden_state = hidden_state + next_input.squeeze(1)

            pred = self.conv_last(hidden_state)
            predictions.append(pred)
            next_input = hidden_state.unsqueeze(1)
        return torch.stack(predictions, dim=1)

# --- Lightning Module for Multitask Grid Models ---
class GridModelLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate, mask, n_targets, trial=None):
        super().__init__(); self.model = model; self.learning_rate = learning_rate; self.trial = trial
        self.criterion = nn.MSELoss(reduction='none'); self.register_buffer('mask', mask); self.validation_step_outputs = []
        self.n_targets = n_targets
        self.register_buffer('task_weights', torch.ones(n_targets))
    def forward(self, x): return self.model(x)
    def _calculate_masked_loss(self, y_hat, y):
        # y_hat shape: (N, T_out, C_out, H, W) | y shape: (N, T_out, C_out, H, W)
        if y_hat.shape[2] != y.shape[2]: # If target channels don't match pred channels
             y_hat = y_hat[:,:,:y.shape[2],:,:] # Align them
        loss = self.criterion(y_hat, y)
        mask_exp = self.mask.unsqueeze(0).unsqueeze(0)
        weighted = loss * mask_exp * self.task_weights.view(1, 1, -1, 1, 1)
        return weighted.sum() / (mask_exp.sum() * y.size(0) * y.size(1) + 1e-9)
    def training_step(self, batch, batch_idx): x, y = batch; return self._calculate_masked_loss(self(x), y)
    def validation_step(self, batch, batch_idx): x, y = batch; loss = self._calculate_masked_loss(self(x), y); self.log('val_loss', loss, on_epoch=True); self.validation_step_outputs.append(loss); return loss
    def predict_step(self, batch, batch_idx, dataloader_idx=0): x, y = batch; return self(x)
    def on_validation_epoch_end(self):
        if not self.validation_step_outputs: return
        avg_loss = torch.stack(self.validation_step_outputs).mean(); self.log('val_rmse', torch.sqrt(avg_loss)); self.validation_step_outputs.clear()
        if self.trial: self.trial.report(avg_loss, self.current_epoch);
        if self.trial and self.trial.should_prune(): raise optuna.exceptions.TrialPruned()
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,  # 10% warmup
                anneal_strategy='cos',
                div_factor=10.0,
                final_div_factor=1e4,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

# --- Custom Dataset for Multitask ConvLSTM ---
class SequenceDatasetConvLSTMMultitask(Dataset):
    def __init__(self, gridded_data, target_indices, n_steps_in, n_steps_out=1):
        self.data = torch.tensor(gridded_data, dtype=torch.float32).permute(0, 3, 1, 2)
        self.target_indices = target_indices; self.n_steps_in = n_steps_in; self.n_steps_out = n_steps_out
    def __len__(self): return self.data.shape[0] - self.n_steps_in - self.n_steps_out
    def __getitem__(self, idx):
        end_idx = idx + self.n_steps_in; out_end_idx = end_idx + self.n_steps_out
        seq_x = self.data[idx:end_idx]; seq_y = self.data[end_idx:out_end_idx, self.target_indices, :, :]
        return seq_x, seq_y

# --- Main Pipeline Class ---
class ConvLSTMMultitaskPipeline:
    def __init__(self, config_path="config.yaml"):
        self.config_path_abs = os.path.abspath(config_path); self.cfg = load_config(self.config_path_abs)
        self.experiment_name = self.cfg.get('project_setup',{}).get('experiment_name','convlstm_multitask_exp')
        self.project_root_for_paths = os.path.dirname(self.config_path_abs); self.run_output_dir = os.path.join(self.project_root_for_paths, 'run_outputs', self.experiment_name)
        self.run_models_dir = os.path.join(self.project_root_for_paths, 'models_saved', self.experiment_name)
        os.makedirs(self.run_output_dir, exist_ok=True); os.makedirs(self.run_models_dir, exist_ok=True)
        self.model = None; self.all_metrics = {}; self.mask = None; self.full_df_raw = None; self.gridded_data = None
        per_loc_preds_dir_name = self.cfg.get('results', {}).get('per_location_predictions_dir', 'per_location_full_predictions')
        self.per_location_predictions_output_dir = os.path.join(self.run_output_dir, per_loc_preds_dir_name)
        os.makedirs(self.per_location_predictions_output_dir, exist_ok=True)

    def _get_abs_path_from_config_value(self, rp): return os.path.abspath(os.path.join(self.project_root_for_paths, rp)) if rp and not os.path.isabs(rp) else rp
    
    def _calculate_masked_metrics(self, actuals, preds, mask):
        metrics_by_target = {}
        target_vars = self.cfg['project_setup']['target_variables']
        mask_bool = mask.bool().to(actuals.device)
        
        # --- FIX: Slice along the correct dimension (channels/targets) ---
        # Shape is (N, T_out, C_out, H, W)
        for i, target_name in enumerate(target_vars):
            actuals_ch = actuals[:, :, i, :, :] # Slice along dim 2 for channels
            preds_ch = preds[:, :, i, :, :]   # Slice along dim 2 for channels
            
            # Squeeze the time-out dimension if it's 1
            if actuals_ch.shape[1] == 1: actuals_ch = actuals_ch.squeeze(1)
            if preds_ch.shape[1] == 1: preds_ch = preds_ch.squeeze(1)

            batch_mask = mask_bool.expand_as(actuals_ch)
            actuals_np = actuals_ch[batch_mask].flatten().cpu().numpy()
            preds_np = preds_ch[batch_mask].flatten().cpu().numpy()
            metrics = {'rmse': mean_squared_error(actuals_np, preds_np), 'mae': mean_absolute_error(actuals_np, preds_np), 'r2': r2_score(actuals_np, preds_np)}
            metrics_by_target[target_name] = metrics
        return metrics_by_target
        # --- END FIX ---

    def _objective(self, trial, train_ds, val_ds, in_channels, pre_seq_len, aft_seq_len, n_targets,num_workers=2):
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
        cfg = self.cfg.get('convlstm_params', {}).get('tuning', {})
        lr = trial.suggest_float('learning_rate', **cfg.get('learning_rate'))
        n_layers = trial.suggest_int('n_layers', **cfg.get('n_layers'))
        hidden_size = trial.suggest_categorical('hidden_dim_size', cfg.get('hidden_dim_size',{}).get('choices', [32, 64]))
        hidden_dim = [hidden_size] * n_layers 
        kernel_size = trial.suggest_categorical('kernel_size', cfg.get('kernel_size',{}).get('choices', [3,5]))
        kernel_size_tuple = (kernel_size, kernel_size) 
        model = EncodingForecastingConvLSTM(in_channels, hidden_dim, kernel_size_tuple, n_layers, pre_seq_len, aft_seq_len, n_targets)
        lightning_model = GridModelLightningModule(model, lr, self.mask, trial=trial, n_targets=n_targets)
        trainer_params = self.cfg.get('convlstm_params', {}).get('trainer', {})
        early_stopping = EarlyStopping(monitor="val_loss", patience=trainer_params.get('patience_for_early_stopping', 5))
        trainer = pl.Trainer(max_epochs=trainer_params.get('max_epochs', 50), callbacks=[early_stopping], logger=False,
                             enable_checkpointing=False, enable_progress_bar=False, accelerator='auto', devices=1)
        try: trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        except optuna.exceptions.TrialPruned: return float('inf')
        return trainer.callback_metrics.get("val_loss", torch.tensor(float('inf'))).item()

    def run_pipeline(self):
        if not PYTORCH_AVAILABLE: return "Failed: Dependencies not found."
        print(f"\n--- Starting ConvLSTM Multitask Global Pipeline ---");
        raw_path = self._get_abs_path_from_config_value(self.cfg.get('data',{}).get('raw_data_path'))
        self.full_df_raw = load_and_prepare_data({'data': {'raw_data_path': raw_path, 'time_column': self.cfg['data']['time_column']}})
        if self.full_df_raw is None: return "Failed: Data Load"
        
        self.gridded_data, mask = create_gridded_data(self.full_df_raw, self.cfg)
        self.mask = torch.tensor(mask, dtype=torch.float32)

        time_steps = self.full_df_raw[self.cfg['data']['time_column']].unique()
        train_end_idx = np.where(time_steps <= np.datetime64(self.cfg['data']['train_end_date']))[0][-1]
        val_end_idx = np.where(time_steps <= np.datetime64(self.cfg['data']['validation_end_date']))[0][-1]
        train_grid, val_grid, test_grid = self.gridded_data[:train_end_idx+1], self.gridded_data[train_end_idx+1:val_end_idx+1], self.gridded_data[val_end_idx+1:]
        
        target_vars = self.cfg['project_setup']['target_variables']
        features_to_grid = self.cfg['data']['features_to_grid']
        target_indices = [features_to_grid.index(t) for t in target_vars]
        
        seq_cfg = self.cfg.get('sequence_params', {}); n_in, n_out = seq_cfg.get('n_steps_in',12), seq_cfg.get('n_steps_out',1)
        
        train_ds = SequenceDatasetConvLSTMMultitask(train_grid, target_indices, n_in, n_out)
        val_ds = SequenceDatasetConvLSTMMultitask(val_grid, target_indices, n_in, n_out)
        test_ds = SequenceDatasetConvLSTMMultitask(test_grid, target_indices, n_in, n_out)
        
        if len(train_ds) == 0 or len(val_ds) == 0: return "Failed: Not enough data for sequences"
        
        batch_size = self.cfg.get('convlstm_params',{}).get('batch_size',8); num_workers = 2 if os.name != 'nt' else 0

        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda t: self._objective(t, train_ds, val_ds, len(features_to_grid), n_in, n_out, len(target_vars)), 
                       n_trials=self.cfg.get('convlstm_params',{}).get('tuning',{}).get('n_trials', 15),num_workers=num_workers, show_progress_bar=True)
        self.best_hyperparams = study.best_trial.params
        print(f"Pipeline: Optuna found best params: {self.best_hyperparams}")
        
        best = self.best_hyperparams
        final_model_base = EncodingForecastingConvLSTM(len(features_to_grid), [best['hidden_dim_size']] * best['n_layers'], (best['kernel_size'],best['kernel_size']), best['n_layers'], n_in, n_out, len(target_vars))
        final_lightning_model = GridModelLightningModule(final_model_base, best['learning_rate'], self.mask, n_targets=len(target_vars))
        full_train_loader = DataLoader(torch.utils.data.ConcatDataset([train_ds, val_ds]), batch_size=self.best_hyperparams["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.best_hyperparams["batch_size"], shuffle=False)
        ckpt_cb = ModelCheckpoint(dirpath=self.run_models_dir, filename="global-convlstm-multitask-best", monitor="val_loss", mode="min")
        trainer_cfg = self.cfg.get('convlstm_params',{}).get('trainer',{})
        final_trainer = pl.Trainer(max_epochs=trainer_cfg.get('max_epochs',50), callbacks=[ckpt_cb], logger=False, 
                                   enable_progress_bar=trainer_cfg.get('enable_progress_bar',True), accelerator='auto', devices=1)
        final_trainer.fit(model=final_lightning_model, train_dataloaders=full_train_loader, val_dataloaders=val_loader)
        
        best_model_path = ckpt_cb.best_model_path
        print(f"Pipeline: Final model training complete. Best model saved at: {best_model_path}")
        self.model = GridModelLightningModule.load_from_checkpoint(best_model_path, model=final_lightning_model.model, learning_rate=final_lightning_model.learning_rate, mask=self.mask, n_targets=len(target_vars))

        self.evaluate_and_save(final_trainer, train_ds, val_ds, test_ds)
        self.predict_on_full_data()
        
        print(f"--- ConvLSTM Multitask Global Pipeline Run Finished ---")
        return self.all_metrics

    def evaluate_and_save(self, trainer, train_dataset, val_dataset, test_dataset):
        print("\n--- Final Model Evaluation ---"); self.all_metrics = {}
        self.model.eval()
        with torch.no_grad():
            for split_name, dataset in [('train', train_dataset), ('validation', val_dataset), ('test', test_dataset)]:
                if len(dataset) > 0:
                    loader = DataLoader(dataset, batch_size=self.best_hyperparams["batch_size"], shuffle=False)
                    y_actual_list, scaled_preds_list = [], []
                    for _, y_batch in loader: y_actual_list.append(y_batch)
                    scaled_preds_list = trainer.predict(self.model, dataloaders=loader)
                    y_actual_grid = torch.cat(y_actual_list).cpu(); scaled_preds_grid = torch.cat(scaled_preds_list).cpu()
                    metrics = self._calculate_masked_metrics(y_actual_grid, scaled_preds_grid, self.mask)
                    self.all_metrics[split_name] = metrics
                    print(f"  Metrics for {split_name.capitalize()} Set:")
                    for target_name, metric_values in metrics.items():
                        print(f"    Target: {target_name} -> RMSE={metric_values['rmse']:.4f}, MAE={metric_values['mae']:.4f}, R2={metric_values['r2']:.4f}")
        
        metrics_filename = self.cfg.get('results',{}).get('metrics_filename', 'global_convlstm_multitask_metrics.json')
        with open(os.path.join(self.run_output_dir, metrics_filename), 'w') as f:
            json.dump(_to_python_type(self.all_metrics), f, indent=4)
        print(f"Pipeline: Evaluation metrics saved to {os.path.join(self.run_output_dir, metrics_filename)}")

    def predict_on_full_data(self):
        print("\nPipeline: Generating predictions on the full raw dataset...")
        if self.model is None or self.gridded_data is None: return None
        
        target_vars = self.cfg['project_setup']['target_variables']
        features_to_grid = self.cfg['data']['features_to_grid']
        target_indices = [features_to_grid.index(t) for t in target_vars]
        seq_params = self.cfg.get('sequence_params', {}); n_in, n_out = seq_params.get('n_steps_in', 12), seq_params.get('n_steps_out', 1)

        full_dataset = SequenceDatasetConvLSTMMultitask(self.gridded_data, target_indices, n_in, n_out)
        if len(full_dataset) == 0: print("Not enough data for full prediction."); return None
        full_loader = DataLoader(full_dataset, batch_size=self.best_hyperparams["batch_size"], shuffle=False)
        
        self.model.eval()
        with torch.no_grad():
            trainer = pl.Trainer(accelerator='auto', devices=1, logger=False)
            predicted_grids_list = trainer.predict(self.model, dataloaders=full_loader)
            predicted_grids = torch.cat(predicted_grids_list).cpu().numpy()

        if predicted_grids.shape[1] == 1: predicted_grids = predicted_grids.squeeze(1) 
        
        print("Pipeline: Un-gridding predictions to create output CSV...")
        time_steps = self.full_df_raw[self.cfg['data']['time_column']].unique()
        pred_start_time_idx = n_in + n_out - 1
        prediction_times = time_steps[pred_start_time_idx : pred_start_time_idx + len(predicted_grids)]

        output_records = []
        valid_pixel_indices = np.argwhere(self.mask.cpu().numpy() == 1)
        
        if 'row_idx' not in self.full_df_raw.columns:
            grid_cfg = self.cfg.get('gridding', {}); fixed_step = grid_cfg.get('fixed_step', 0.5)
            lat_min = self.full_df_raw[self.cfg['data']['lat_column']].min(); lon_min = self.full_df_raw[self.cfg['data']['lon_column']].min()
            self.full_df_raw['row_idx'] = ((self.full_df_raw[self.cfg['data']['lat_column']] - lat_min) / fixed_step).round().astype(int)
            self.full_df_raw['col_idx'] = ((self.full_df_raw[self.cfg['data']['lon_column']] - lon_min) / fixed_step).round().astype(int)
        
        cell_to_coord = self.full_df_raw[['row_idx','col_idx','lat','lon']].drop_duplicates().set_index(['row_idx','col_idx'])

        for i, t in enumerate(tqdm(prediction_times, desc="Un-gridding predictions")):
            for r, c in valid_pixel_indices:
                record = {'time': t}
                try: coords = cell_to_coord.loc[(r,c)]; record['lat'], record['lon'] = coords.lat, coords.lon
                except KeyError: continue
                
                actual_value_row = self.full_df_raw[(self.full_df_raw['time'] == t) & (self.full_df_raw['lat'] == record['lat']) & (self.full_df_raw['lon'] == record['lon'])]
                
                for target_idx, target_name in enumerate(target_vars):
                    record[target_name] = actual_value_row[target_name].values[0] if not actual_value_row.empty else np.nan
                    record[f'{target_name}_predicted'] = predicted_grids[i, target_idx, r, c]
                
                output_records.append(record)

        output_df = pd.DataFrame(output_records)
        pred_filename = self.cfg.get('results',{}).get('predictions_filename', 'global_convlstm_multitask_full_predictions.csv')
        save_path = os.path.join(self.run_output_dir, pred_filename)
        output_df.to_csv(save_path, index=False)
        print(f"Pipeline: Full data predictions saved to {save_path}")
