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
    from torch.utils.data import TensorDataset, DataLoader
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    import optuna
    # PyTorchLightningPruningCallback is no longer needed with this new approach
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
    print("LSTM PyTorch Lightning Pipeline: Successfully imported utility functions.")
except ImportError as e:
    print(f"LSTM PyTorch Lightning Pipeline Class Error: Could not import utility functions: {e}")

# --- PyTorch LSTM Model Definition ---
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x); last_time_step_out = lstm_out[:, -1, :]; out = self.fc(last_time_step_out)
        return out

# --- PyTorch Lightning Module (MODIFIED FOR OPTUNA INTEGRATION & PREDICTION) ---
class LSTMLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate, trial=None): # Added trial as an optional parameter
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.trial = trial # Store the optuna trial
        self.criterion = nn.MSELoss()
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch; y_hat = self(x); loss = self.criterion(y_hat, y)
        self.log('train_loss', loss); return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch; y_hat = self(x); loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append(loss)
        return loss
    
    # --- NEW METHOD to fix prediction error ---
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # The batch from DataLoader is a list [features, targets]
        # We only need the features for prediction.
        x, y = batch # Unpack the batch
        return self(x) # Call forward with only the features tensor
    # --- END NEW METHOD ---

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs: return

        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('val_rmse', torch.sqrt(avg_loss))
        self.validation_step_outputs.clear()

        # If we are in a tuning trial, report to Optuna and check for pruning
        if self.trial:
            self.trial.report(avg_loss, self.current_epoch)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# --- Sequence Generation (same as before) ---
def create_sequences_torch(features_df, target_series, n_steps_in, n_steps_out=1):
    X, y = [], [];
    for i in range(len(features_df) - n_steps_in - n_steps_out + 1):
        X.append(features_df.iloc[i:i + n_steps_in].values); y.append(target_series.iloc[i + n_steps_in:i + n_steps_in + n_steps_out].values)
    if not X: return torch.empty(0), torch.empty(0)
    X_np, y_np = np.array(X), np.array(y)
    if n_steps_out == 1: y_np = y_np.reshape((len(y_np), 1))
    return torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32)

# --- Main Pipeline Class ---
class LSTMPyTorchLightningLocalPipeline:
    def __init__(self, config_path="config.yaml"):
        # (Initialization is the same)
        self.config_path_abs = os.path.abspath(config_path); self.cfg = load_config(self.config_path_abs)
        self.experiment_name = self.cfg.get('project_setup',{}).get('experiment_name','lstm_pytorch_lightning_exp')
        self.project_root_for_paths = os.path.dirname(self.config_path_abs)
        results_base_cfg = self.cfg.get('results',{}).get('output_base_dir','run_outputs')
        self.run_output_dir = os.path.join(self.project_root_for_paths, results_base_cfg, self.experiment_name)
        models_base_dir_cfg = self.cfg.get('paths',{}).get('models_base_dir','models_saved')
        self.run_models_dir_base = os.path.join(self.project_root_for_paths, models_base_dir_cfg, self.experiment_name)
        per_loc_preds_dir_name = self.cfg.get('results',{}).get('per_location_predictions_dir','per_location_full_predictions')
        self.per_location_predictions_output_dir = os.path.join(self.run_output_dir, per_loc_preds_dir_name)
        os.makedirs(self.run_output_dir, exist_ok=True); os.makedirs(self.run_models_dir_base, exist_ok=True); os.makedirs(self.per_location_predictions_output_dir, exist_ok=True)
        print(f"Pipeline artifacts will be saved under subdirectories of '{self.run_output_dir}' and '{self.run_models_dir_base}'")
        self.full_raw_data = None; self.unique_locations = []; self.all_location_metrics = []

    def _get_abs_path_from_config_value(self, relative_path): # (Helper function)
        if not relative_path or os.path.isabs(relative_path): return relative_path
        return os.path.abspath(os.path.join(self.project_root_for_paths, relative_path))
        
    def _load_full_data(self): # (Same as before)
        print("Pipeline: Loading full raw data..."); raw_path = self.cfg.get('data',{}).get('raw_data_path'); abs_path = self._get_abs_path_from_config_value(raw_path)
        if not raw_path or not abs_path or not os.path.exists(abs_path): return False
        temp_cfg = {'data': {'raw_data_path': abs_path, 'time_column': self.cfg['data']['time_column']}}; self.full_raw_data = load_and_prepare_data(temp_cfg)
        if self.full_raw_data is None: return False
        lat_col, lon_col = self.cfg['data']['lat_column'], self.cfg['data']['lon_column']
        self.unique_locations = self.full_raw_data[[lat_col, lon_col]].drop_duplicates().values.tolist()
        print(f"Pipeline: Found {len(self.unique_locations)} unique locations."); return True

    def _calculate_metrics(self, actuals, predictions): # (Same as before)
        rmse = mean_squared_error(actuals, predictions); mae = mean_absolute_error(actuals, predictions); r2 = r2_score(actuals, predictions)
        return {'rmse': rmse, 'mae': mae, 'r2': r2}

    def _objective_for_optuna(self, trial, train_loader, val_loader, n_features, n_steps_in, n_steps_out):
        # (This objective function remains the same as it's generic)
        lstm_tuning_cfg = self.cfg.get('lstm_params', {}).get('tuning', {})
        n_layers = trial.suggest_int('n_layers', **lstm_tuning_cfg.get('n_layers', {'low':1, 'high':3}))
        hidden_size = trial.suggest_int('hidden_size', **lstm_tuning_cfg.get('hidden_size', {'low':32, 'high':128, 'step':16}))
        dropout_rate = trial.suggest_float('dropout_rate', **lstm_tuning_cfg.get('dropout_rate', {'low':0.1, 'high':0.5}))
        learning_rate = trial.suggest_float('learning_rate', **lstm_tuning_cfg.get('learning_rate', {'low':1e-4, 'high':1e-2, 'log':True}))
        model = LSTMRegressor(n_features, hidden_size, n_layers, n_steps_out, dropout_rate)
        lightning_model = LSTMLightningModule(model, learning_rate, trial=trial)
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=self.cfg.get('lstm_params',{}).get('trainer',{}).get('patience_for_early_stopping', 5))
        trainer_params = self.cfg.get('lstm_params', {}).get('trainer', {})
        trainer = pl.Trainer(max_epochs=50, callbacks=[early_stopping_callback], logger=False,
            enable_checkpointing=False, enable_progress_bar=trainer_params.get('enable_progress_bar', False), accelerator=trainer_params.get('accelerator', 'auto'), devices=1)
        try:
            trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        except optuna.exceptions.TrialPruned:
            return float('inf')
        return trainer.callback_metrics.get("val_loss", torch.tensor(float('inf'))).item()

    def _process_location(self, location_coords, location_data):
        # (This method remains largely the same until final training)
        lat, lon = location_coords; loc_identifier = f"lat{lat}_lon{lon}"
        print(f"\n--- Processing Location: {loc_identifier} ---")

        train_df_raw, val_df_raw, test_df_raw = split_data_chronologically(location_data.copy(), self.cfg)
        if train_df_raw is None or train_df_raw.empty: return None
        train_df_featured = self._add_time_features(train_df_raw.copy())
        val_df_featured = self._add_time_features(val_df_raw.copy())
        test_df_featured = self._add_time_features(test_df_raw.copy())

        if train_df_featured.empty: return None
        scaled_train_df, scaled_val_df, scaled_test_df, fitted_scaler = scale_data(train_df_featured, val_df_featured, test_df_featured, self.cfg)
        if fitted_scaler is None: return None

        target_col = self.cfg['project_setup']['target_variable']
        cols_to_drop = [self.cfg['data']['time_column'], self.cfg['data']['lat_column'], self.cfg['data']['lon_column']]
        X_train_flat = scaled_train_df.drop(columns=cols_to_drop, errors='ignore'); y_train_flat = X_train_flat.pop(target_col)
        X_val_flat = scaled_val_df.drop(columns=cols_to_drop, errors='ignore'); y_val_flat = X_val_flat.pop(target_col)
        X_test_flat = scaled_test_df.drop(columns=cols_to_drop, errors='ignore'); y_test_flat = X_test_flat.pop(target_col)
        print(X_train_flat.head())
        
        lstm_params = self.cfg.get('lstm_params', {})
        n_steps_in = lstm_params.get('n_steps_in', 12); n_steps_out = lstm_params.get('n_steps_out', 1)
        X_train_seq, y_train_seq = create_sequences_torch(X_train_flat, y_train_flat, n_steps_in, n_steps_out)
        X_val_seq, y_val_seq = create_sequences_torch(X_val_flat, y_val_flat, n_steps_in, n_steps_out)
        X_test_seq, y_test_seq = create_sequences_torch(X_test_flat, y_test_flat, n_steps_in, n_steps_out)
        
        if X_train_seq.shape[0] == 0 or X_val_seq.shape[0] == 0: return None
        
        batch_size = lstm_params.get('batch_size', 32)
        train_loader = DataLoader(TensorDataset(X_train_seq, y_train_seq), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_seq, y_val_seq), batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(X_test_seq, y_test_seq), batch_size=batch_size) 
        
        n_features = X_train_seq.shape[2]
        n_trials = self.cfg.get('lstm_params', {}).get('tuning', {}).get('n_trials', 15)
        print(f"  Starting Optuna hyperparameter optimization for {n_trials} trials...")
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial: self._objective_for_optuna(trial, train_loader, val_loader, n_features, n_steps_in, n_steps_out), n_trials=n_trials)
        best_hyperparams = study.best_trial.params
        print(f"  Optuna found best params: {best_hyperparams}")

        print("  Training final model with best hyperparameters...")
        final_model_base = LSTMRegressor(n_features, best_hyperparams['hidden_size'], best_hyperparams['n_layers'], n_steps_out, best_hyperparams['dropout_rate'])
        final_lightning_model = LSTMLightningModule(final_model_base, best_hyperparams['learning_rate'])
        
        full_train_loader = DataLoader(TensorDataset(torch.cat([X_train_seq, X_val_seq]), torch.cat([y_train_seq, y_val_seq])), batch_size=batch_size, shuffle=True)
        
        local_model_dir = os.path.join(self.run_models_dir_base, "local_lstm_models")
        os.makedirs(local_model_dir, exist_ok=True)
        
        checkpoint_callback = ModelCheckpoint(dirpath=local_model_dir, filename=f"{loc_identifier}-best-model", save_top_k=1, verbose=False, monitor="val_loss", mode="min")
        trainer_params = self.cfg.get('lstm_params', {}).get('trainer', {})
        final_trainer = pl.Trainer(max_epochs=trainer_params.get('max_epochs', 50), callbacks=[checkpoint_callback], logger=False,
            enable_progress_bar=trainer_params.get('enable_progress_bar', False), accelerator=trainer_params.get('accelerator', 'auto'), devices=1)
        final_trainer.fit(model=final_lightning_model, train_dataloaders=full_train_loader, val_dataloaders=val_loader)
        best_model_path = checkpoint_callback.best_model_path
        print(f"  Final model training complete. Best model saved at: {best_model_path}")
        
        best_lightning_model = LSTMLightningModule.load_from_checkpoint(best_model_path, model=final_lightning_model.model, learning_rate=final_lightning_model.learning_rate)
        
        location_metrics_summary = {'location': loc_identifier, 'lat': lat, 'lon': lon, 'best_params': best_hyperparams}
        best_lightning_model.eval();
        with torch.no_grad():
            for split_name, loader in [('train', train_loader), ('validation', val_loader), ('test', test_loader)]:
                if len(loader.dataset) > 0:
                    scaled_predictions = torch.cat(final_trainer.predict(best_lightning_model, dataloaders=loader)).numpy()
                    y_actual_seq = loader.dataset.tensors[1].numpy()
                    inversed_preds = inverse_transform_predictions(pd.DataFrame(scaled_predictions, columns=[target_col]), target_col, fitted_scaler)
                    inversed_actuals = inverse_transform_predictions(pd.DataFrame(y_actual_seq, columns=[target_col]), target_col, fitted_scaler)
                    if inversed_preds is not None and inversed_actuals is not None:
                        metrics = self._calculate_metrics(inversed_actuals, inversed_preds); location_metrics_summary[split_name] = metrics
                        print(f"  {loc_identifier} - {split_name.capitalize()} Set: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")
        
        save_scaler(fitted_scaler, os.path.join(local_model_dir, f"{loc_identifier}_scaler.joblib"))
        self._predict_and_save_full_for_location(location_data, loc_identifier, best_lightning_model.model, fitted_scaler)

        return location_metrics_summary
    def _add_time_features(self, df):
        time_col = self.cfg['data']['time_column']
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df["year"] = df[time_col].dt.year
            df["month"] = df[time_col].dt.month
        return df
    def _predict_and_save_full_for_location(self, location_data_raw_df, loc_identifier, model, scaler):
        # (This method remains the same)
        print(f"  Generating full predictions for {loc_identifier}...")
        full_featured_df = self._add_time_features(location_data_raw_df.copy())
        if full_featured_df.empty: return
        target_col = self.cfg['project_setup']['target_variable']; cols_to_scale = [target_col] + self.cfg.get('data', {}).get('predictor_columns', [])
        actual_cols_to_scale = [col for col in cols_to_scale if col in full_featured_df.columns]
        scaled_full_featured = full_featured_df.copy(); scaled_full_featured[actual_cols_to_scale] = scaler.transform(full_featured_df[actual_cols_to_scale])
        cols_to_drop = [self.cfg['data']['time_column'], self.cfg['data']['lat_column'], self.cfg['data']['lon_column']]
        X_flat = scaled_full_featured.drop(columns=cols_to_drop, errors='ignore'); y_flat = X_flat.pop(target_col)
        n_steps_in = self.cfg.get('lstm_params', {}).get('n_steps_in', 12); n_steps_out = self.cfg.get('lstm_params', {}).get('n_steps_out', 1)
        X_seq_full, _ = create_sequences_torch(X_flat, y_flat, n_steps_in, n_steps_out)
        if X_seq_full.shape[0] == 0: return
        model.eval()
        with torch.no_grad(): scaled_predictions = model(X_seq_full).numpy()
        dummy_df_preds = pd.DataFrame(scaled_predictions, columns=[target_col]); inversed_predictions = inverse_transform_predictions(dummy_df_preds, target_col, scaler)
        pred_indices = full_featured_df.index[n_steps_in + n_steps_out - 1:]
        output_df = location_data_raw_df.copy(); output_df[f'{target_col}_predicted'] = pd.Series(inversed_predictions.values.flatten(), index=pred_indices)
        cols_to_save = [self.cfg['data']['time_column'], target_col, f'{target_col}_predicted']
        filename_suffix = self.cfg.get('results',{}).get('per_location_prediction_filename_suffix', '_full_pred.csv')
        save_path = os.path.join(self.per_location_predictions_output_dir, f"{loc_identifier}{filename_suffix}")
        output_df[cols_to_save].to_csv(save_path, index=False); print(f"    Full predictions for {loc_identifier} saved.")
    
    def run_pipeline(self):
        # (This method remains the same)
        if not PYTORCH_AVAILABLE: print("Cannot run LSTM pipeline."); return "Failed: PyTorch/Lightning/Optuna not found."
        print(f"\n--- Starting LSTM PyTorch Lightning Local Pipeline ---");
        if not self._load_full_data(): print("Pipeline Halted: Failed at data loading."); return "Failed: Data Load"
        self.all_location_metrics = []
        lat_col, lon_col = self.cfg['data']['lat_column'], self.cfg['data']['lon_column']
        for loc_coords in self.unique_locations:
            location_data_df = self.full_raw_data[(self.full_raw_data[lat_col] == loc_coords[0]) & (self.full_raw_data[lon_col] == loc_coords[1])]
            if location_data_df.empty: continue
            metrics = self._process_location(loc_coords, location_data_df)
            if metrics: self.all_location_metrics.append(metrics)
        self._save_aggregated_metrics()
        print(f"--- LSTM PyTorch Lightning Local Pipeline Run Finished ---")
        return self.all_location_metrics
    
    def _save_aggregated_metrics(self):
        # (This method remains the same)
        if not self.all_location_metrics: return
        metrics_filename = self.cfg.get('results',{}).get('metrics_filename', 'lstm_pytorch_local_metrics.json')
        metrics_save_path = os.path.join(self.run_output_dir, metrics_filename)
        try:
            with open(metrics_save_path, 'w') as f: json.dump(self.all_location_metrics, f, indent=4)
            print(f"Pipeline: Aggregated metrics saved to {metrics_save_path}")
        except Exception as e: print(f"Pipeline Error: Could not save aggregated metrics: {e}")
