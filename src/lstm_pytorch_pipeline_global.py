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
    print("LSTM Global Pipeline: Successfully imported utility functions.")
except ImportError as e:
    print(f"LSTM Global Pipeline Error: Could not import utility functions: {e}")

# --- Reusable Components (can be shared across pipelines) ---
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x); last_time_step_out = lstm_out[:, -1, :]; out = self.fc(last_time_step_out)
        return out

class LSTMLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate, trial=None):
        super().__init__(); self.model = model; self.learning_rate = learning_rate
        self.trial = trial; self.criterion = nn.MSELoss(); self.validation_step_outputs = []
    def forward(self, x): return self.model(x)
    def training_step(self, batch, batch_idx):
        x, y = batch; y_hat = self(x); loss = self.criterion(y_hat, y)
        self.log('train_loss', loss); return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch; y_hat = self(x); loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append(loss); return loss
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch; return self(x)
    def on_validation_epoch_end(self):
        if not self.validation_step_outputs: return
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('val_rmse', torch.sqrt(avg_loss)); self.validation_step_outputs.clear()
        if self.trial:
            self.trial.report(avg_loss, self.current_epoch)
            if self.trial.should_prune(): raise optuna.exceptions.TrialPruned()
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# --- NEW: MEMORY-EFFICIENT CUSTOM DATASET ---
class SequenceDataset(Dataset):
    def __init__(self, features_df, target_series, n_steps_in, n_steps_out=1):
        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        self.target = torch.tensor(target_series.values, dtype=torch.float32)
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out

    def __len__(self):
        return len(self.features) - self.n_steps_in - self.n_steps_out + 1

    def __getitem__(self, idx):
        end_idx = idx + self.n_steps_in
        out_end_idx = end_idx + self.n_steps_out
        
        seq_x = self.features[idx:end_idx]
        seq_y = self.target[end_idx:out_end_idx]

        if self.n_steps_out == 1:
            return seq_x, seq_y.squeeze() # Return a scalar if only one step
        
        return seq_x, seq_y

# --- GLOBAL LSTM PIPELINE CLASS ---
class LSTMPyTorchGlobalPipeline:
    def __init__(self, config_path="config.yaml"):
        # (Initialization is the same, setting up paths)
        self.config_path_abs = os.path.abspath(config_path); self.cfg = load_config(self.config_path_abs)
        self.experiment_name = self.cfg.get('project_setup',{}).get('experiment_name','lstm_global_experiment')
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

    # --- Main Pipeline Logic (No Location Loop) ---
    def run_pipeline(self):
        if not PYTORCH_AVAILABLE: return "Failed: PyTorch/Lightning/Optuna not found."
        print(f"\n--- Starting LSTM PyTorch GLOBAL Pipeline ---");

        # 1. Load and Split Data
        print("Pipeline: Loading and splitting data...")
        raw_path = self.cfg.get('data',{}).get('raw_data_path'); abs_path = self._get_abs_path_from_config_value(raw_path)
        if not raw_path or not abs_path or not os.path.exists(abs_path): return "Failed: Data Load"
        temp_cfg = {'data': {'raw_data_path': abs_path, 'time_column': self.cfg['data']['time_column']}}
        full_df_raw = load_and_prepare_data(temp_cfg)
        if full_df_raw is None: return "Failed: Data Load"
        self.full_df_raw_for_prediction = full_df_raw.copy()
        train_df_raw, val_df_raw, test_df_raw = split_data_chronologically(full_df_raw, self.cfg)
        if train_df_raw is None or train_df_raw.empty: return "Failed: Data Split resulted in empty train set."

        # 2. Feature Engineering on full splits
        print("Pipeline: Engineering features...")
        train_df_featured = engineer_features(train_df_raw.copy(), self.cfg)
        val_df_featured = engineer_features(val_df_raw.copy(), self.cfg)
        test_df_featured = engineer_features(test_df_raw.copy(), self.cfg)
        if train_df_featured.empty: return "Failed: Feature engineering resulted in empty train set."

        # 3. Scale Data with a SINGLE global scaler
        print("Pipeline: Scaling data...")
        scaled_train_df, scaled_val_df, scaled_test_df, fitted_scaler = scale_data(train_df_featured, val_df_featured, test_df_featured, self.cfg)
        if fitted_scaler is None: return "Failed: Scaling"
        self.scaler = fitted_scaler
        
        # 4. Create Datasets and DataLoaders (MEMORY EFFICIENT)
        print("Pipeline: Creating Datasets and DataLoaders...")
        target_col = self.cfg['project_setup']['target_variable']
        cols_to_drop = [self.cfg['data']['time_column']] 
        X_train_flat = scaled_train_df.drop(columns=cols_to_drop, errors='ignore'); y_train_flat = X_train_flat.pop(target_col)
        X_val_flat = scaled_val_df.drop(columns=cols_to_drop, errors='ignore'); y_val_flat = X_val_flat.pop(target_col)
        X_test_flat = scaled_test_df.drop(columns=cols_to_drop, errors='ignore'); y_test_flat = X_test_flat.pop(target_col)
        
        lstm_params = self.cfg.get('lstm_params', {})
        n_steps_in = lstm_params.get('n_steps_in', 12); n_steps_out = lstm_params.get('n_steps_out', 1)
        
        train_dataset = SequenceDataset(X_train_flat, y_train_flat, n_steps_in, n_steps_out)
        val_dataset = SequenceDataset(X_val_flat, y_val_flat, n_steps_in, n_steps_out)
        test_dataset = SequenceDataset(X_test_flat, y_test_flat, n_steps_in, n_steps_out)

        if len(train_dataset) == 0 or len(val_dataset) == 0: return "Failed: Not enough data to create sequences."
        
        batch_size = lstm_params.get('batch_size', 256)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, persistent_workers=True)
        
        # 5. Hyperparameter Tuning
        n_features = X_train_flat.shape[1]
        n_trials = self.cfg.get('lstm_params', {}).get('tuning', {}).get('n_trials', 15)
        print(f"Pipeline: Starting Optuna for {n_trials} trials...")
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial: self._objective_for_optuna(trial, train_loader, val_loader, n_features, n_steps_in, n_steps_out), n_trials=n_trials)
        self.best_hyperparams = study.best_trial.params
        print(f"Pipeline: Optuna found best params: {self.best_hyperparams}")

        # 6. Train Final Model
        print("Pipeline: Training final model...")
        final_model_base = LSTMRegressor(n_features, self.best_hyperparams['hidden_size'], self.best_hyperparams['n_layers'], n_steps_out, self.best_hyperparams['dropout_rate'])
        final_lightning_model = LSTMLightningModule(final_model_base, self.best_hyperparams['learning_rate'])
        full_train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
        full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        checkpoint_callback = ModelCheckpoint(dirpath=self.run_models_dir, filename="global-lstm-best-model", save_top_k=1, verbose=False, monitor="val_loss", mode="min")
        trainer_params = self.cfg.get('lstm_params', {}).get('trainer', {})
        final_trainer = pl.Trainer(max_epochs=trainer_params.get('max_epochs', 50), callbacks=[checkpoint_callback], logger=False,
            enable_progress_bar=trainer_params.get('enable_progress_bar', True), accelerator=trainer_params.get('accelerator', 'auto'), devices=1)
        final_trainer.fit(model=final_lightning_model, train_dataloaders=full_train_loader, val_dataloaders=val_loader)
        best_model_path = checkpoint_callback.best_model_path
        print(f"Pipeline: Final model training complete. Best model saved at: {best_model_path}")
        self.model = LSTMLightningModule.load_from_checkpoint(best_model_path, model=final_lightning_model.model, learning_rate=final_lightning_model.learning_rate)
        
        # 7. Evaluate and Save
        self.evaluate_and_save(final_trainer, fitted_scaler, train_loader, val_loader, DataLoader(test_dataset, batch_size=batch_size, num_workers=2))
        self.predict_on_full_data()
        
        print(f"--- LSTM Global Pipeline Run Finished ---")
        return self.all_metrics

    def evaluate_and_save(self, trainer, scaler, train_loader, val_loader, test_loader):
        print("\n--- Final Model Evaluation ---")
        target_col = self.cfg['project_setup']['target_variable']
        self.model.eval()
        with torch.no_grad():
            for split_name, loader in [('train', train_loader), ('validation', val_loader), ('test', test_loader)]:
                if len(loader.dataset) > 0:
                    # Get raw tensors from dataset for actuals
                    if isinstance(loader.dataset, torch.utils.data.ConcatDataset):
                        y_actual_seq = torch.cat([d.target for d in loader.dataset.datasets])
                    else:
                        y_actual_seq = loader.dataset.target
                    
                    # Get predictions
                    scaled_preds = torch.cat(trainer.predict(self.model, dataloaders=loader)).numpy()
                    
                    # Align actuals with predictions (predictions will be shorter due to windowing)
                    n_steps_in = self.cfg.get('lstm_params',{}).get('n_steps_in',12)
                    n_steps_out = self.cfg.get('lstm_params',{}).get('n_steps_out',1)
                    start_idx = n_steps_in + n_steps_out - 1
                    y_actual_aligned = y_actual_seq[start_idx : start_idx + len(scaled_preds)]

                    inversed_preds = inverse_transform_predictions(pd.DataFrame(scaled_preds, columns=[target_col]), target_col, scaler)
                    inversed_actuals = inverse_transform_predictions(pd.DataFrame(y_actual_aligned.numpy(), columns=[target_col]), target_col, scaler)
                    
                    if inversed_preds is not None and inversed_actuals is not None:
                        metrics = self._calculate_metrics(inversed_actuals, inversed_preds)
                        self.all_metrics[split_name] = metrics
                        print(f"  {split_name.capitalize()} Set: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")
        
        metrics_filename = self.cfg.get('results',{}).get('metrics_filename', 'global_lstm_metrics.json')
        with open(os.path.join(self.run_output_dir, metrics_filename), 'w') as f: json.dump(self.all_metrics, f, indent=4)
        print(f"Pipeline: Evaluation metrics saved.")
        scaler_filename = self.cfg.get('scaling', {}).get('scaler_filename', 'global_scaler.joblib')
        save_scaler(scaler, os.path.join(self.run_models_dir, scaler_filename))
        print(f"Pipeline: Global scaler saved.")

    def _objective_for_optuna(self, trial, train_loader, val_loader, n_features, n_steps_in, n_steps_out):
        # (Same as before)
        lstm_tuning_cfg = self.cfg.get('lstm_params', {}).get('tuning', {})
        n_layers = trial.suggest_int('n_layers', **lstm_tuning_cfg.get('n_layers')); hidden_size = trial.suggest_int('hidden_size', **lstm_tuning_cfg.get('hidden_size'))
        dropout_rate = trial.suggest_float('dropout_rate', **lstm_tuning_cfg.get('dropout_rate')); learning_rate = trial.suggest_float('learning_rate', **lstm_tuning_cfg.get('learning_rate'))
        model = LSTMRegressor(n_features, hidden_size, n_layers, n_steps_out, dropout_rate); lightning_model = LSTMLightningModule(model, learning_rate, trial=trial)
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=self.cfg.get('lstm_params',{}).get('trainer',{}).get('patience_for_early_stopping', 5))
        trainer_params = self.cfg.get('lstm_params', {}).get('trainer', {})
        trainer = pl.Trainer(max_epochs=trainer_params.get('max_epochs', 50), callbacks=[early_stopping_callback], logger=False,
            enable_checkpointing=False, enable_progress_bar=trainer_params.get('enable_progress_bar', False), accelerator=trainer_params.get('accelerator', 'auto'), devices=1)
        try: trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        except optuna.exceptions.TrialPruned: return float('inf')
        return trainer.callback_metrics.get("val_loss", torch.tensor(float('inf'))).item()
        
    def predict_on_full_data(self):
        # (This method also needs to use the new Dataset)
        print("\nPipeline: Generating predictions on the full raw dataset...")
        if self.model is None or self.scaler is None: return None
        
        full_featured_df = engineer_features(self.full_df_raw_for_prediction.copy(), self.cfg)
        if full_featured_df.empty: return None
        
        target_col = self.cfg['project_setup']['target_variable']; cols_to_scale = [target_col] + self.cfg['data']['predictor_columns']
        actual_cols_to_scale = [col for col in cols_to_scale if col in full_featured_df.columns]
        scaled_full_featured = full_featured_df.copy(); scaled_full_featured[actual_cols_to_scale] = self.scaler.transform(full_featured_df[actual_cols_to_scale])
        
        cols_to_drop = [self.cfg['data']['time_column']]; X_flat = scaled_full_featured.drop(columns=cols_to_drop, errors='ignore'); y_flat = X_flat.pop(target_col)
        
        n_steps_in = self.cfg.get('lstm_params', {}).get('n_steps_in', 12); n_steps_out = self.cfg.get('lstm_params', {}).get('n_steps_out', 1)
        full_dataset = SequenceDataset(X_flat, y_flat, n_steps_in, n_steps_out)
        
        if len(full_dataset) == 0: return None
        
        full_loader = DataLoader(full_dataset, batch_size=self.cfg.get('lstm_params',{}).get('batch_size',256))
        
        self.model.eval()
        with torch.no_grad():
            trainer_for_pred = pl.Trainer(accelerator=self.cfg.get('lstm_params',{}).get('trainer',{}).get('accelerator','auto'), devices=1, logger=False)
            scaled_predictions = torch.cat(trainer_for_pred.predict(self.model, dataloaders=full_loader)).numpy()
        
        inversed_predictions = inverse_transform_predictions(pd.DataFrame(scaled_predictions, columns=[target_col]), target_col, self.scaler)
        
        pred_indices = full_featured_df.index[n_steps_in + n_steps_out - 1:]
        output_df = self.full_df_raw_for_prediction.copy()
        output_df[f'{target_col}_predicted'] = pd.Series(inversed_predictions.values.flatten(), index=pred_indices)
        
        pred_filename = self.cfg.get('results',{}).get('predictions_filename', 'global_lstm_full_predictions.csv')
        save_path = os.path.join(self.run_output_dir, pred_filename)
        output_df[[self.cfg['data']['time_column'], 'lat', 'lon', target_col, f'{target_col}_predicted']].to_csv(save_path, index=False)
        print(f"Pipeline: Full data predictions saved to {save_path}")
