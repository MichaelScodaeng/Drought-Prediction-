import pandas as pd
import numpy as np
import yaml
import os
import joblib
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import json
import optuna
# import matplotlib.pyplot as plt # Not strictly needed for this class functionality

# Assuming these are in src/ or PYTHONPATH is set for the notebook
try:
    from src.data_utils import load_config, load_and_prepare_data, split_data_chronologically
    from src.preprocess_utils import scale_data, save_scaler, load_scaler, inverse_transform_predictions
    from src.feature_utils import engineer_features
    print("Local Pipeline Class: Successfully imported utility functions.")
except ImportError as e:
    print(f"Local Pipeline Class Error: Could not import utility functions: {e}")
    # Define dummy functions if import fails, so class can be parsed
    def load_config(path=None): return {}
    def load_and_prepare_data(config=None): return None
    def split_data_chronologically(df=None, config=None): return None, None, None
    def engineer_features(df=None, config=None): return df
    def scale_data(df_train=None, df_val=None, df_test=None, config=None): return None,None,None,None
    def save_scaler(scaler=None, path=None): pass
    def load_scaler(path=None): return None
    def inverse_transform_predictions(df=None, target=None, scaler=None): return None


class XGBoostLocalPipeline:
    def __init__(self, config_path="config.yaml"):
        self.config_path_abs = os.path.abspath(config_path)
        print(f"Local Pipeline Class: Attempting to load config from: {self.config_path_abs}")
        self.cfg = load_config(self.config_path_abs)
        
        if not self.cfg or self.cfg.get('data',{}).get('raw_data_path') is None:
            print("Local Pipeline Warning: Configuration might not have loaded correctly.")

        self.experiment_name = self.cfg.get('project_setup', {}).get('experiment_name', 'local_xgb_experiment')
        self.project_root_for_paths = os.path.dirname(self.config_path_abs)

        results_base_cfg = self.cfg.get('paths',{}).get('output_base_dir', 'run_outputs')
        self.run_output_dir = os.path.join(self.project_root_for_paths, results_base_cfg, self.experiment_name + "_local")
        
        models_base_dir_cfg = self.cfg.get('paths', {}).get('models_base_dir', 'models_saved')
        self.run_models_dir_base = os.path.join(self.project_root_for_paths, models_base_dir_cfg, self.experiment_name + "_local")
        print("self.run_models_dir_base, ", self.run_models_dir_base)
        print("run_output_dir, ", self.run_output_dir)

        # Directory for per-location full predictions
        per_loc_preds_dir_name = self.cfg.get('paths', {}).get('per_location_predictions_dir', 'per_location_full_predictions')
        self.per_location_predictions_output_dir = os.path.join(self.run_output_dir, per_loc_preds_dir_name)


        os.makedirs(self.run_output_dir, exist_ok=True)
        os.makedirs(self.run_models_dir_base, exist_ok=True) 
        os.makedirs(self.per_location_predictions_output_dir, exist_ok=True) # Create this dir

        print(f"Local Pipeline Class: Artifacts for experiment '{self.experiment_name}_local' will be saved under various subdirectories of '{self.run_output_dir}' and '{self.run_models_dir_base}'")

        self.all_location_metrics = []
        self.full_raw_data = None
        self.unique_locations = []

    def _get_abs_path_from_config_value(self, relative_path_from_config_value):
        if relative_path_from_config_value is None: return None
        if os.path.isabs(relative_path_from_config_value): return relative_path_from_config_value
        return os.path.abspath(os.path.join(self.project_root_for_paths, relative_path_from_config_value))
    
    def _remove_non_lag_features(self, df, location_id="Unknown"):
        """
        Removes features that do not contain '_lag' to prevent data leakage.
        Keeps time, target, and coordinates intact.
        """
        time_col = self.cfg['data']['time_column']
        target_col = self.cfg['project_setup']['target_variable']
        lat_col = self.cfg.get('data', {}).get('lat_column', 'lat')
        lon_col = self.cfg.get('data', {}).get('lon_column', 'lon')

        safe_cols = [time_col, target_col, lat_col, lon_col]
        lag_cols = [col for col in df.columns if '_lag' in col]
        keep_cols = list(set(lag_cols + safe_cols))

        dropped_cols = [col for col in df.columns if col not in keep_cols and col!="month" and col!="year"]
        if dropped_cols:
            print(f"  [Leakage Prevention] {location_id}: Dropping non-lag features: {dropped_cols}")

        return df[keep_cols].copy()
    def _tune_xgboost_model(self, X_train, y_train, X_val, y_val):
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'tree_method': 'hist',
            }
            model = xgb.XGBRegressor(**params, early_stopping_rounds=15)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            return root_mean_squared_error(y_val, preds)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.cfg.get('model_params', {}).get('optuna_trials', 20))
        print(f"    [Optuna] Best hyperparameters: {study.best_params}")
        return study.best_params
    
    def _load_full_data(self):
        print("Local Pipeline: Loading full raw data...")
        relative_raw_data_path = self.cfg.get('data', {}).get('raw_data_path')
        if not relative_raw_data_path:
            print("Local Pipeline Error: 'data.raw_data_path' not found in configuration.")
            return False
        abs_data_file_path = self._get_abs_path_from_config_value(relative_raw_data_path)
        if not abs_data_file_path or not os.path.exists(abs_data_file_path):
            print(f"Local Pipeline Error: Data file not found at: {abs_data_file_path}")
            return False
        
        temp_load_cfg = self.cfg.copy(); temp_load_cfg['data'] = self.cfg['data'].copy() 
        temp_load_cfg['data']['raw_data_path'] = abs_data_file_path 
        self.full_raw_data = load_and_prepare_data(temp_load_cfg) 
        if self.full_raw_data is None:
            print("Local Pipeline Error: Failed to load full_raw_data."); return False
        
        lat_col = self.cfg.get('data', {}).get('lat_column', 'lat')
        lon_col = self.cfg.get('data', {}).get('lon_column', 'lon')
        if lat_col in self.full_raw_data.columns and lon_col in self.full_raw_data.columns:
            self.unique_locations = self.full_raw_data[[lat_col, lon_col]].drop_duplicates().values.tolist()
            print(f"Local Pipeline: Found {len(self.unique_locations)} unique locations.")
        else:
            print("Local Pipeline Error: Latitude/Longitude columns not found for identifying unique locations.")
            return False
        return True

    def _predict_and_save_full_for_location(self, location_data_raw_df, loc_identifier, local_model, local_scaler, model_input_columns):
        """Generates and saves predictions for the entire time series of a single location."""
        print(f"  Generating full predictions for {loc_identifier}...")
        if local_model is None or local_scaler is None:
            print(f"    Skipping full prediction for {loc_identifier}: model or scaler missing.")
            return

        # 1. Engineer features on the full raw data for this location
        location_data_featured = location_data_raw_df.sort_values(by=self.cfg['data']['time_column']).copy()  # Sort by time column
        engineered = engineer_features(location_data_raw_df.copy(), self.cfg)
        location_data_featured = self._remove_non_lag_features(engineered, loc_identifier)

        if location_data_featured.empty:
            print(f"    Skipping full prediction for {loc_identifier}: feature engineering resulted in empty data.")
            return

        target_col_name = self.cfg['project_setup']['target_variable']
         # 2. Prepare      features for scaling (columns scaler was fitted on)
        scaler_feature_names = list(local_scaler.feature_names_in_) if hasattr(local_scaler, 'feature_names_in_') else []
        if not scaler_feature_names:
            print(f"    Skipping full prediction for {loc_identifier}: local scaler has no feature_names_in_."); return

        df_to_scale_local_full = pd.DataFrame(index=location_data_featured.index)
        for col in scaler_feature_names:
            if col in location_data_featured:
                df_to_scale_local_full[col] = location_data_featured[col]
            else:
                df_to_scale_local_full[col] = np.nan 
        
        # 3. Scale the features
        scaled_subset_df_local = df_to_scale_local_full.copy() # Avoid SettingWithCopy
        cols_present_and_to_scale = [col for col in scaler_feature_names if col in df_to_scale_local_full.columns]

        if not cols_present_and_to_scale:
            print(f"   Skipping full prediction for {loc_identifier}: No columns to scale found in prepared data matching scaler's features.")
            return
        
        try:
            scaled_subset_df_local[cols_present_and_to_scale] = local_scaler.transform(df_to_scale_local_full[cols_present_and_to_scale])
        except Exception as e:
            print(f"   Error scaling full data for {loc_identifier}: {e}. Skipping full prediction.")
            return


        # 4. Prepare X for prediction (matching columns model was trained on)
        X_local_full_for_prediction = pd.DataFrame(index=location_data_featured.index)
        #print(X_local_full_for_prediction.shape, "Shape of X_local_full_for_prediction before adding columns")
        #print(X_local_full_for_prediction.tail())  # Print first few rows for debugging
        for col in model_input_columns: # model_input_columns are from X_train for this location
            if col in scaled_subset_df_local.columns: # If it was a column that got scaled by local_scaler
                X_local_full_for_prediction[col] = scaled_subset_df_local[col]
            elif col in location_data_featured.columns: # If it's an unscaled feature (like month, year if not in predictor_cols)
                X_local_full_for_prediction[col] = location_data_featured[col]
            else:
                print(f"    Warning for {loc_identifier}: Feature '{col}' expected by local model not found in processed full data. Filling with 0.")
                X_local_full_for_prediction[col] = 0 
        
        # 5. Make predictions
        scaled_predictions_local_full = local_model.predict(X_local_full_for_prediction)

        # 6. Inverse transform predictions
        scaled_preds_local_full_df = pd.DataFrame(scaled_predictions_local_full, columns=[target_col_name], index=location_data_featured.index)
        inversed_predictions_local_full = inverse_transform_predictions(scaled_preds_local_full_df, target_col_name, local_scaler)
        #print(f"    Inversed predictions for {loc_identifier} generated successfully.")
        #print(inversed_predictions_local_full.head())  # Print first few rows for debugging
        # 7. Combine with original relevant columns and save
        if inversed_predictions_local_full is not None:
            # Start with the raw location data, then merge predictions
            output_df = location_data_raw_df.copy()
            # Add predictions where indices match (from location_data_featured after dropna)
            output_df = output_df.merge(
                pd.DataFrame({f'{target_col_name}_predicted': inversed_predictions_local_full}, index=location_data_featured.index),
                left_index=True,
                right_index=True,
                how='left'
            )
            
            cols_to_save = [self.cfg['data']['time_column'], target_col_name, f'{target_col_name}_predicted']
            # Add some original predictors for context
            for orig_pred_col in ['pre','tmp']: 
                if orig_pred_col in output_df.columns: cols_to_save.append(orig_pred_col)
            
            final_output_df_subset = output_df[[col for col in cols_to_save if col in output_df.columns]]

            filename_suffix = self.cfg.get('results',{}).get('per_location_prediction_filename_suffix', '_full_pred.csv')
            save_filename = f"{loc_identifier}{filename_suffix}"
            save_path = os.path.join(self.per_location_predictions_output_dir, save_filename)
            try:
                final_output_df_subset.to_csv(save_path, index=False)
                print(f"    Full predictions for {loc_identifier} saved to {save_path}")
            except Exception as e:
                print(f"    Error saving full predictions for {loc_identifier}: {e}")
        else:
            print(f"    Failed to inverse transform full predictions for {loc_identifier}.")


    def _process_location(self, location_coords, location_data_raw_df):
        lat, lon = location_coords
        loc_identifier = f"lat{lat}_lon{lon}"
        print(f"\n--- Processing Location: {loc_identifier} ---")

        train_df_raw, val_df_raw, test_df_raw = split_data_chronologically(location_data_raw_df.copy(), self.cfg) # Pass copy
        if train_df_raw is None or train_df_raw.empty:
            print(f"Warning: No training data for {loc_identifier} after split. Skipping this location.")
            return None

        train_df_featured = self._remove_non_lag_features(engineer_features(train_df_raw.copy(), self.cfg), loc_identifier)
        val_df_featured = self._remove_non_lag_features(engineer_features(val_df_raw.copy(), self.cfg), loc_identifier)
        test_df_featured = self._remove_non_lag_features(engineer_features(test_df_raw.copy(), self.cfg), loc_identifier)
        
        if train_df_featured.empty:
            print(f"Warning: Training data empty after feature engineering for {loc_identifier}. Skipping.")
            return None

        scaled_train_df, scaled_val_df, scaled_test_df, fitted_scaler = scale_data(
            train_df_featured, val_df_featured, test_df_featured, self.cfg
        )
        if fitted_scaler is None:
            print(f"Warning: Scaling failed for {loc_identifier}. Skipping.")
            return None
        
        local_scaler_dir = os.path.join(self.run_models_dir_base, "local_scalers")
        os.makedirs(local_scaler_dir, exist_ok=True)
        scaler_filename_base = self.cfg.get('scaling',{}).get('scaler_filename', 'local_robust_scaler.joblib')
        scaler_save_path = os.path.join(local_scaler_dir, f"{loc_identifier}_{scaler_filename_base}")
        save_scaler(fitted_scaler, scaler_save_path)

        target_col = self.cfg['project_setup']['target_variable']
        time_col = self.cfg['data']['time_column']
        lat_col_cfg = self.cfg.get('data', {}).get('lat_column', 'lat')
        lon_col_cfg = self.cfg.get('data', {}).get('lon_column', 'lon')
        
        cols_to_drop_for_X = list(set([target_col, time_col, lat_col_cfg, lon_col_cfg]))


        X_train = scaled_train_df.drop(columns=cols_to_drop_for_X, errors='ignore')
        y_train = scaled_train_df[target_col]
        X_val = scaled_val_df.drop(columns=cols_to_drop_for_X, errors='ignore')
        y_val = scaled_val_df[target_col]
        X_test = scaled_test_df.drop(columns=cols_to_drop_for_X, errors='ignore')
        y_test = scaled_test_df[target_col]
        
        if X_train.empty or y_train.empty:
            print(f"Warning: X_train or y_train is empty for {loc_identifier}. Skipping.")
            return None

        model_input_columns_for_this_loc = X_train.columns.tolist() # Store for full prediction

        print(f"  Tuning XGBoost hyperparameters for {loc_identifier}...")
        best_params = self._tune_xgboost_model(X_train, y_train, X_val, y_val)
        model = xgb.XGBRegressor(**best_params, early_stopping_rounds=10)
        fit_params = {'verbose': False}
        if xgb.__version__ >= '0.90' and not X_val.empty and not y_val.empty :
             fit_params['eval_set'] = [(X_val, y_val)]
        try: model.fit(X_train, y_train, **fit_params)
        except Exception as e: print(f"Error training model for {loc_identifier}: {e}"); return None

        local_model_dir = os.path.join(self.run_models_dir_base, "local_models")
        os.makedirs(local_model_dir, exist_ok=True)
        
        model_params_xgb_config = self.cfg.get('model_params', {}).get('global_xgboost', {})
        model_filename_base = model_params_xgb_config.get('model_filename', 'xgboost_model.json')
        model_save_path = os.path.join(local_model_dir, f"{loc_identifier}_{model_filename_base}")
        try: model.save_model(model_save_path)
        except Exception as e: print(f"Error saving model for {loc_identifier}: {e}")

        location_metrics_summary = {'location': loc_identifier, 'lat': lat, 'lon': lon}
        for split_name, X_eval, y_eval_scaled_local in [('train', X_train, y_train),('validation', X_val, y_val), ('test', X_test, y_test)]:
            if X_eval is None or X_eval.empty or y_eval_scaled_local is None or y_eval_scaled_local.empty:
                location_metrics_summary[split_name] = None; continue
            scaled_predictions = model.predict(X_eval)
            scaled_actuals_df = pd.DataFrame(y_eval_scaled_local.values, columns=[target_col], index=y_eval_scaled_local.index)
            scaled_preds_df = pd.DataFrame(scaled_predictions, columns=[target_col], index=y_eval_scaled_local.index)
            inversed_predictions = inverse_transform_predictions(scaled_preds_df, target_col, fitted_scaler)
            inversed_actuals = inverse_transform_predictions(scaled_actuals_df, target_col, fitted_scaler)
            if inversed_predictions is not None and inversed_actuals is not None:
                metrics = {'rmse': root_mean_squared_error(inversed_actuals, inversed_predictions), 
                           'mae': mean_absolute_error(inversed_actuals, inversed_predictions), 
                           'r2': r2_score(inversed_actuals, inversed_predictions)}
                print(f"  {loc_identifier} - {split_name.capitalize()} Set: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")
                location_metrics_summary[split_name] = metrics
            else: location_metrics_summary[split_name] = None
        
        # Generate and save full predictions for this location
        self._predict_and_save_full_for_location(location_data_raw_df, loc_identifier, model, fitted_scaler, model_input_columns_for_this_loc)
        location_metrics_summary['best_params'] = best_params
        return location_metrics_summary

    def run_pipeline(self):
        #print(f"\n--- Starting Local XGBoost Pipeline Run: Experiment '{self.experiment_name}_local' ---")
        if not self._load_full_data():
            print("Pipeline Halted: Failed at initial full data loading."); return "Failed: Data Load"
        
        self.all_location_metrics = []
        lat_col = self.cfg.get('data', {}).get('lat_column', 'lat')
        lon_col = self.cfg.get('data', {}).get('lon_column', 'lon')

        for loc_coords in self.unique_locations:
            current_lat, current_lon = loc_coords
            location_data_df = self.full_raw_data[
                (self.full_raw_data[lat_col] == current_lat) &
                (self.full_raw_data[lon_col] == current_lon)
            ].copy()
            if location_data_df.empty: continue
            metrics = self._process_location(loc_coords, location_data_df)
            if metrics: self.all_location_metrics.append(metrics)
        
        self._save_aggregated_metrics()
        #print(f"--- Local XGBoost Pipeline Run Finished: Experiment '{self.experiment_name}_local' ---")
           
        return self.all_location_metrics

    def _save_aggregated_metrics(self):
        if not self.all_location_metrics:
            print("No metrics collected from local models to save."); return
        metrics_filename = self.cfg.get('results',{}).get('metrics_filename', 'local_evaluation_metrics.json')
        metrics_save_path = os.path.join(self.run_output_dir, metrics_filename)
        try:
            with open(metrics_save_path, 'w') as f:
                json.dump(self.all_location_metrics, f, indent=4)
            print(f"Local Pipeline: Aggregated evaluation metrics saved to {metrics_save_path}")
        except Exception as e: print(f"Local Pipeline Error: Could not save aggregated metrics: {e}")

