import pandas as pd
import numpy as np
import yaml
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

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
    from src.data_utils import load_config, load_and_prepare_data, split_data_chronologically
    from src.preprocess_utils import scale_data, save_scaler, load_scaler, inverse_transform_predictions
    from src.feature_utils import engineer_features
    print("CNN1D Global Pipeline: Successfully imported utility functions.")
except ImportError as e:
    print(f"CNN1D Global Pipeline Error: Could not import utility functions: {e}")

# --- NEW PyTorch CNN1D Model Definition ---
class CNN1DRegressor(nn.Module):
    def __init__(self, n_features, n_conv_layers, out_channels, kernel_size, dropout_rate, n_steps_out):
        super(CNN1DRegressor, self).__init__()
        layers = []
        in_channels = n_features
        
        for i in range(n_conv_layers):
            layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'))
            layers.append(nn.ReLU())
            in_channels = out_channels # The output of one layer is the input to the next
        
        layers.append(nn.Flatten())
        
        self.conv_block = nn.Sequential(*layers)
        self.fc = None # Will be created dynamically
        self.dropout = nn.Dropout(dropout_rate)
        self.n_steps_out = n_steps_out

    def forward(self, x):
        # Conv1d expects (batch_size, n_features, seq_len), so we permute
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        
        if self.fc is None:
            self.fc = nn.Linear(x.shape[1], self.n_steps_out).to(x.device)
        
        x = self.dropout(x)
        out = self.fc(x)
        return out

# --- Generic Lightning Module (reusable for LSTM or CNN) ---
class SequenceModelLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate, trial=None):
        super().__init__(); self.model = model; self.learning_rate = learning_rate
        self.trial = trial; self.criterion = nn.MSELoss(); self.validation_step_outputs = []
    def forward(self, x): return self.model(x)
    def training_step(self, batch, batch_idx):
        x, y = batch; y_hat = self(x); loss = self.criterion(y_hat, y)
        self.log('train_loss', loss); return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch; y_hat = self(x); loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True); self.validation_step_outputs.append(loss); return loss
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch; return self(x)
    def on_validation_epoch_end(self):
        if not self.validation_step_outputs: return
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('val_rmse', torch.sqrt(avg_loss)); self.validation_step_outputs.clear()
        if self.trial: self.trial.report(avg_loss, self.current_epoch);
        if self.trial and self.trial.should_prune(): raise optuna.exceptions.TrialPruned()
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# --- Sequence Dataset (MODIFIED) ---
class SequenceDataset(Dataset):
    def __init__(self, features_df, target_series, group_by_cols, n_steps_in, n_steps_out=1):
        self.features_np = torch.tensor(features_df.drop(columns=group_by_cols).values, dtype=torch.float32)
        
        # --- FIX: Ensure target is always a 2D tensor ---
        target_vals = target_series.values
        if target_vals.ndim == 1:
            target_vals = target_vals.reshape(-1, 1) # Reshape 1D array to 2D
        self.target_np = torch.tensor(target_vals, dtype=torch.float32)
        # --- END FIX ---

        self.n_steps_in = n_steps_in; self.n_steps_out = n_steps_out; self.indices = []
        group_ids = features_df[group_by_cols].apply(tuple, axis=1)
        group_change_indices = np.where(group_ids.values[:-1] != group_ids.values[1:])[0] + 1
        group_starts = np.insert(group_change_indices, 0, 0)
        group_ends = np.append(group_starts[1:], len(features_df))
        for start, end in zip(group_starts, group_ends):
            num_sequences = (end - start) - n_steps_in - n_steps_out + 1
            if num_sequences > 0: self.indices.extend(range(start, start + num_sequences))

    def __len__(self): return len(self.indices)
    
    def __getitem__(self, idx):
        start_pos = self.indices[idx]; end_pos = start_pos + self.n_steps_in; out_end_pos = end_pos + self.n_steps_out
        seq_x = self.features_np[start_pos:end_pos]
        seq_y = self.target_np[end_pos:out_end_pos]

        # --- FIX: Ensure output shape is consistent ---
        # If n_steps_out is 1, seq_y will have shape [1, 1]. Squeeze it to [1]
        # to match the model's final linear layer output shape of [batch_size, 1]
        # when the dataloader batches it.
        if self.n_steps_out == 1:
            return seq_x, seq_y.squeeze(0)
        # --- END FIX ---
        
        return seq_x, seq_y

# --- GLOBAL CNN1D PIPELINE CLASS ---
class CNN1DGlobalPipeline:
    def __init__(self, config_path="config.yaml"):
        # (Initialization is the same, setting up paths)
        self.config_path_abs = os.path.abspath(config_path); self.cfg = load_config(self.config_path_abs)
        self.experiment_name = self.cfg.get('project_setup',{}).get('experiment_name','cnn1d_global_experiment')
        self.project_root_for_paths = os.path.dirname(self.config_path_abs)
        results_base_cfg = self.cfg.get('results',{}).get('output_base_dir','run_outputs')
        self.run_output_dir = os.path.join(self.project_root_for_paths, results_base_cfg, self.experiment_name)
        models_base_dir_cfg = self.cfg.get('paths',{}).get('models_base_dir','models_saved')
        self.run_models_dir = os.path.join(self.project_root_for_paths, models_base_dir_cfg, self.experiment_name)
        os.makedirs(self.run_output_dir, exist_ok=True); os.makedirs(self.run_models_dir, exist_ok=True)
        print(f"Pipeline artifacts will be saved under '{self.run_output_dir}' and '{self.run_models_dir}'")
        self.scaler = None; self.model = None; self.all_metrics = {}

    def _get_abs_path_from_config_value(self, relative_path): # (Helper function)
        if not relative_path or os.path.isabs(relative_path): return relative_path
        return os.path.abspath(os.path.join(self.project_root_for_paths, relative_path))
    def _calculate_metrics(self, actuals, predictions): # (Helper function)
        rmse = mean_squared_error(actuals, predictions, squared=False); mae = mean_absolute_error(actuals, predictions); r2 = r2_score(actuals, predictions)
        return {'rmse': rmse, 'mae': mae, 'r2': r2}

    def _objective_for_optuna(self, trial, train_loader, val_loader, n_features, n_steps_in, n_steps_out):
        cnn_tuning_cfg = self.cfg.get('cnn1d_params', {}).get('tuning', {})
        learning_rate = trial.suggest_float('learning_rate', **cnn_tuning_cfg.get('learning_rate'))
        n_conv_layers = trial.suggest_int('n_conv_layers', **cnn_tuning_cfg.get('n_conv_layers'))
        out_channels_power = trial.suggest_int('out_channels_power', **cnn_tuning_cfg.get('out_channels_power'))
        out_channels = 2**out_channels_power
        kernel_size = trial.suggest_categorical('kernel_size', cnn_tuning_cfg.get('kernel_size', {}).get('choices', [2, 3]))
        dropout_rate = trial.suggest_float('dropout_rate', **cnn_tuning_cfg.get('dropout_rate'))
        
        model = CNN1DRegressor(n_features, n_conv_layers, out_channels, kernel_size, dropout_rate, n_steps_out)
        lightning_model = SequenceModelLightningModule(model, learning_rate, trial=trial)
        
        trainer_params = self.cfg.get('cnn1d_params', {}).get('trainer', {})
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=trainer_params.get('patience_for_early_stopping', 5))
        trainer = pl.Trainer(max_epochs=trainer_params.get('max_epochs', 50), callbacks=[early_stopping_callback],
            logger=False, enable_checkpointing=False, enable_progress_bar=False, accelerator='auto', devices=1)
        try:
            trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        except optuna.exceptions.TrialPruned: return float('inf')
        return trainer.callback_metrics.get("val_loss", torch.tensor(float('inf'))).item()

    def run_pipeline(self):
        if not PYTORCH_AVAILABLE: return "Failed: PyTorch/Lightning/Optuna not found."
        print(f"\n--- Starting CNN1D PyTorch GLOBAL Pipeline ---");
        
        # 1. Load, sort, and split data (same as global LSTM)
        print("Pipeline: Loading, sorting, and splitting data...")
        raw_path = self.cfg.get('data',{}).get('raw_data_path'); abs_path = self._get_abs_path_from_config_value(raw_path)
        if not raw_path or not abs_path or not os.path.exists(abs_path): return "Failed: Data Load"
        temp_cfg = {'data': {'raw_data_path': abs_path, 'time_column': self.cfg['data']['time_column']}}
        full_df_raw = load_and_prepare_data(temp_cfg)
        if full_df_raw is None: return "Failed: Data Load"
        sort_cols = [self.cfg['data']['lat_column'], self.cfg['data']['lon_column'], self.cfg['data']['time_column']]
        full_df_raw.sort_values(by=sort_cols, inplace=True); full_df_raw.reset_index(drop=True, inplace=True)
        train_df_raw, val_df_raw, test_df_raw = split_data_chronologically(full_df_raw, self.cfg)
        if train_df_raw is None or train_df_raw.empty: return "Failed: Data Split"

        # 2. Feature engineering (same as global LSTM)
        print("Pipeline: Engineering features..."); train_df_featured = engineer_features(train_df_raw.copy(), self.cfg)
        val_df_featured = engineer_features(val_df_raw.copy(), self.cfg); test_df_featured = engineer_features(test_df_raw.copy(), self.cfg)
        if train_df_featured.empty: return "Failed: Feature Engineering"

        # 3. Scaling (same as global LSTM)
        print("Pipeline: Scaling data..."); scaled_train_df, scaled_val_df, scaled_test_df, fitted_scaler = scale_data(train_df_featured, val_df_featured, test_df_featured, self.cfg)
        if fitted_scaler is None: return "Failed: Scaling"
        self.scaler = fitted_scaler

        # 4. Create Datasets and DataLoaders (same as global LSTM)
        print("Pipeline: Creating Datasets and DataLoaders...")
        target_col = self.cfg['project_setup']['target_variable']; feature_cols = [self.cfg['data']['lat_column'], self.cfg['data']['lon_column'], target_col] + self.cfg['data']['predictor_columns']
        feature_cols_exist = [col for col in feature_cols if col in scaled_train_df.columns]
        X_train_flat = scaled_train_df[feature_cols_exist]; y_train_flat = X_train_flat.pop(target_col)
        X_val_flat = scaled_val_df[feature_cols_exist]; y_val_flat = X_val_flat.pop(target_col)
        X_test_flat = scaled_test_df[feature_cols_exist]; y_test_flat = X_test_flat.pop(target_col)
        
        cnn_params = self.cfg.get('cnn1d_params', {})
        n_steps_in = cnn_params.get('n_steps_in', 12); n_steps_out = cnn_params.get('n_steps_out', 1)
        group_by_cols_for_seq = [self.cfg['data']['lat_column'], self.cfg['data']['lon_column']]
        train_dataset = SequenceDataset(X_train_flat, y_train_flat, group_by_cols_for_seq, n_steps_in, n_steps_out)
        val_dataset = SequenceDataset(X_val_flat, y_val_flat, group_by_cols_for_seq, n_steps_in, n_steps_out)
        test_dataset = SequenceDataset(X_test_flat, y_test_flat, group_by_cols_for_seq, n_steps_in, n_steps_out)
        if len(train_dataset) == 0 or len(val_dataset) == 0: return "Failed: Not enough data for sequences"
        
        batch_size = cnn_params.get('batch_size', 256); num_workers = 2 if os.name != 'nt' else 0
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

        # 5. Hyperparameter Tuning
        n_features = train_dataset.features_np.shape[1]
        n_trials = self.cfg.get('cnn1d_params', {}).get('tuning', {}).get('n_trials', 15)
        print(f"Pipeline: Starting Optuna for {n_trials} trials... (Model input_size will be {n_features})")
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial: self._objective_for_optuna(trial, train_loader, val_loader, n_features, n_steps_in, n_steps_out), n_trials=n_trials)
        self.best_hyperparams = study.best_trial.params
        print(f"Pipeline: Optuna found best params: {self.best_hyperparams}")

        # 6. Train Final Model
        print("Pipeline: Training final model..."); best = self.best_hyperparams
        final_model_base = CNN1DRegressor(n_features, best['n_conv_layers'], 2**best['out_channels_power'], best['kernel_size'], best['dropout_rate'], n_steps_out)
        final_lightning_model = SequenceModelLightningModule(final_model_base, best['learning_rate'])
        full_train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
        full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        checkpoint_callback = ModelCheckpoint(dirpath=self.run_models_dir, filename="global-cnn1d-best-model", save_top_k=1, verbose=False, monitor="val_loss", mode="min")
        trainer_params = self.cfg.get('cnn1d_params', {}).get('trainer', {})
        final_trainer = pl.Trainer(max_epochs=trainer_params.get('max_epochs', 50), callbacks=[checkpoint_callback], logger=False,
            enable_progress_bar=trainer_params.get('enable_progress_bar', True), accelerator='auto', devices=1)
        final_trainer.fit(model=final_lightning_model, train_dataloaders=full_train_loader, val_dataloaders=val_loader)
        best_model_path = checkpoint_callback.best_model_path
        print(f"Pipeline: Final model training complete. Best model saved at: {best_model_path}")
        self.model = SequenceModelLightningModule.load_from_checkpoint(best_model_path, model=final_lightning_model.model, learning_rate=final_lightning_model.learning_rate)
        
        # 7. Evaluate and Save
        # (This part can be encapsulated in a helper method if needed, but is here for clarity)
        print("\n--- Final Model Evaluation ---"); self.all_metrics = {}
        self.model.eval()
        with torch.no_grad():
            for split_name, loader in [('train', train_loader), ('validation', val_loader), ('test', DataLoader(test_dataset, batch_size=batch_size))]:
                if len(loader.dataset) > 0:
                    scaled_preds = torch.cat(final_trainer.predict(self.model, dataloaders=loader)).numpy()
                    
                    if isinstance(loader.dataset, torch.utils.data.ConcatDataset):
                        y_actual_seq = torch.cat([d.target_np for d in loader.dataset.datasets])
                        # For ConcatDataset, indices need careful handling if they are not contiguous
                        # This part might need adjustment if using ConcatDataset for evaluation
                        # A simpler way is to just predict on the original train and val loaders
                        # But for now, we assume indices align for the sake of the example
                    else:
                         y_actual_seq = loader.dataset.target_np[loader.dataset.indices]

                    inversed_preds = inverse_transform_predictions(pd.DataFrame(scaled_preds, columns=[target_col]), target_col, fitted_scaler)
                    inversed_actuals = inverse_transform_predictions(pd.DataFrame(y_actual_seq.numpy(), columns=[target_col]), target_col, fitted_scaler)
                    
                    if inversed_preds is not None and inversed_actuals is not None:
                        # Ensure lengths match for metric calculation
                        min_len = min(len(inversed_preds), len(inversed_actuals))
                        metrics = self._calculate_metrics(inversed_actuals[:min_len], inversed_preds[:min_len])
                        self.all_metrics[split_name] = metrics
                        print(f"  {split_name.capitalize()} Set: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")
        
        metrics_filename = self.cfg.get('results',{}).get('metrics_filename', 'global_cnn1d_metrics.json')
        with open(os.path.join(self.run_output_dir, metrics_filename), 'w') as f: json.dump(self.all_metrics, f, indent=4)
        print(f"Pipeline: Evaluation metrics saved."); scaler_filename = self.cfg.get('scaling', {}).get('scaler_filename', 'global_scaler_cnn1d.joblib')
        save_scaler(fitted_scaler, os.path.join(self.run_models_dir, scaler_filename)); print(f"Pipeline: Global scaler saved.")
        
        print(f"--- CNN1D Global Pipeline Run Finished ---")
        return self.all_metrics

