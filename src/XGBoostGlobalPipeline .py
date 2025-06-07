from matplotlib import pyplot as plt
import os
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_utils import load_config, load_and_prepare_data, split_data_chronologically, engineer_features, scale_data, save_scaler, inverse_transform_predictions
class XGBoostGlobalPipeline:
    def __init__(self, config_path="config.yaml"):
        # config_path should be relative to the CWD of the script/notebook calling this,
        # or an absolute path.
        self.config_path_abs = os.path.abspath(config_path)
        print(f"Pipeline Class: Attempting to load config from: {self.config_path_abs}")
        self.cfg = load_config(self.config_path_abs) # load_config from data_utils handles fallback
        
        if not self.cfg or self.cfg.get('data',{}).get('raw_data_path') is None : # Check if default was used and critical path is missing
            # This check might be too simple if DEFAULT_CONFIG in data_utils is well-defined
            # A better check is if self.cfg IS the DEFAULT_CONFIG object identity from data_utils
            print("Pipeline Class Warning: Configuration might not have loaded correctly or is missing critical paths. Check config path and content.")

        self.scaler = None
        self.model = None
        self.best_hyperparams = None
        self.train_df_raw, self.val_df_raw, self.test_df_raw = None, None, None
        self.train_df_featured, self.val_df_featured, self.test_df_featured = None, None, None
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = [None]*6

    def _get_abs_path_from_config_value(self, relative_path_from_config_value):
        """Constructs an absolute path from a path value read from the config file.
        The path value is assumed to be relative to the location of the config file itself.
        """
        if relative_path_from_config_value is None:
            return None
        if os.path.isabs(relative_path_from_config_value):
            return relative_path_from_config_value
        
        # project_root_for_paths is the directory containing the config file
        project_root_for_paths = os.path.dirname(self.config_path_abs)
        abs_path = os.path.abspath(os.path.join(project_root_for_paths, relative_path_from_config_value))
        return abs_path

    def load_and_split_data(self):
        print("Pipeline: Loading and splitting data...")
        
        relative_raw_data_path = self.cfg.get('data', {}).get('raw_data_path')
        if not relative_raw_data_path:
            print("Pipeline Error: 'data.raw_data_path' not found in configuration.")
            # So self.train_df_raw etc. will remain None
            return

        abs_data_file_path = self._get_abs_path_from_config_value(relative_raw_data_path)

        if not abs_data_file_path or not os.path.exists(abs_data_file_path):
            print(f"Pipeline Error: Data file not found at constructed absolute path: {abs_data_file_path}")
            print(f" (Derived from config file location '{self.config_path_abs}' and config path value '{relative_raw_data_path}')")
            return

        # Create a temporary config for load_and_prepare_data,
        # ensuring it receives the absolute path in the key it expects.
        temp_load_cfg = self.cfg.copy() 
        temp_load_cfg['data'] = self.cfg['data'].copy() 
        temp_load_cfg['data']['raw_data_path'] = abs_data_file_path # Pass the absolute path

        full_df_raw = load_and_prepare_data(temp_load_cfg) 

        if full_df_raw is None:
            print("Pipeline Error: data_utils.load_and_prepare_data returned None. Check its logs.")
            return

        self.train_df_raw, self.val_df_raw, self.test_df_raw = split_data_chronologically(full_df_raw, self.cfg)
        print("Pipeline: Data loaded and split.")
        if self.train_df_raw is None or self.train_df_raw.empty:
             print("Pipeline Warning: train_df_raw is None or empty after splitting.")
        else:
             print(f"Pipeline: train_df_raw shape: {self.train_df_raw.shape}")


    def engineer_all_features(self):
        print("Pipeline: Engineering features...")
        if self.train_df_raw is None: # Check if data was loaded
            print("Pipeline Error: Raw training data not loaded. Cannot engineer features.")
            raise ValueError("Raw training data not loaded for feature engineering.")
        
        self.train_df_featured = engineer_features(self.train_df_raw.copy(), self.cfg)
        self.val_df_featured = engineer_features(self.val_df_raw.copy(), self.cfg)
        self.test_df_featured = engineer_features(self.test_df_raw.copy(), self.cfg)
        print("Pipeline: Feature engineering complete.")
        if self.train_df_featured is None or self.train_df_featured.empty:
            print("Pipeline Warning: train_df_featured is None or empty after engineering.")


    def preprocess_all_data(self):
        print("Pipeline: Scaling data...")
        if self.train_df_featured is None:
            print("Pipeline Error: Featured training data not available. Cannot scale.")
            raise ValueError("Featured training data not available for scaling.")
            
        scaled_train, scaled_val, scaled_test, fitted_sclr = scale_data(
            self.train_df_featured, self.val_df_featured, self.test_df_featured, self.cfg
        )
        if fitted_sclr is None:
            print("Pipeline Error: scale_data did not return a fitted scaler.")
            raise ValueError("Scaler fitting failed.")

        self.scaler = fitted_sclr
        
        target_col = self.cfg['project_setup']['target_variable']
        time_col = self.cfg['data']['time_column']
        cols_to_drop_for_X = [target_col]
        if time_col in scaled_train.columns: cols_to_drop_for_X.append(time_col)

        self.X_train = scaled_train.drop(columns=cols_to_drop_for_X, errors='ignore')
        self.y_train = scaled_train[target_col]
        self.X_val = scaled_val.drop(columns=cols_to_drop_for_X, errors='ignore')
        self.y_val = scaled_val[target_col]
        self.X_test = scaled_test.drop(columns=cols_to_drop_for_X, errors='ignore')
        self.y_test = scaled_test[target_col]

        scaler_path_cfg = self.cfg.get('scaling',{}).get('scaler_path')
        if scaler_path_cfg:
            scaler_save_path = self._get_abs_path_from_config_value(scaler_path_cfg)
            if scaler_save_path:
                save_scaler(self.scaler, scaler_save_path)
        print("Pipeline: Data scaling and X,y preparation complete.")

    def _objective_for_optuna(self, trial):
        target_col = self.cfg['project_setup']['target_variable']
        # Define search space for hyperparameters
        param = {
            'objective': self.cfg.get('model_params', {}).get('global_xgboost', {}).get('objective', 'reg:squarederror'),
            'eval_metric': self.cfg.get('model_params', {}).get('global_xgboost', {}).get('eval_metric', 'rmse'),
            'tree_method': 'hist',
            'random_state': self.cfg.get('project_setup', {}).get('random_seed', 42),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        }
        model = xgb.XGBRegressor(**param,early_stopping_rounds = 10)
        fit_params_opt = {'verbose': False}
        if xgb.__version__ >= '0.90': # Check XGBoost version for early_stopping_rounds in fit
             fit_params_opt['eval_set'] = [(self.X_val, self.y_val)] # Use self.X_val, self.y_val
        
        model.fit(self.X_train, self.y_train, **fit_params_opt) # Use self.X_train, self.y_train
        preds_val_scaled = model.predict(self.X_val) # Use self.X_val
        
        scaled_preds_val_df_opt = pd.DataFrame(preds_val_scaled, columns=[target_col], index=self.X_val.index)
        inversed_predictions_val_opt = inverse_transform_predictions(scaled_preds_val_df_opt, target_col, self.scaler)
        
        scaled_actuals_val_df_opt = pd.DataFrame(self.y_val.values, columns=[target_col], index=self.y_val.index)
        inversed_actuals_val_opt = inverse_transform_predictions(scaled_actuals_val_df_opt, target_col, self.scaler)

        if inversed_predictions_val_opt is None or inversed_actuals_val_opt is None: return float('inf')
        
        rmse = mean_squared_error(inversed_actuals_val_opt, inversed_predictions_val_opt)
        return rmse

    def tune_hyperparameters(self, n_trials=50):
        print("Pipeline: Tuning hyperparameters...")
        if self.X_train is None:
            print("Pipeline Error: Data not preprocessed. Cannot tune.")
            raise ValueError("Data not preprocessed for hyperparameter tuning.")
            
        study = optuna.create_study(direction='minimize')
        study.optimize(self._objective_for_optuna, n_trials=n_trials)
        self.best_hyperparams = study.best_trial.params
        print(f"Pipeline: Hyperparameter tuning complete. Best RMSE on validation: {study.best_trial.value:.4f}")
        print(f"Best params: {self.best_hyperparams}")

    def train_final_model(self, params=None):
        print("Pipeline: Training final model...")
        if self.X_train is None:
            print("Pipeline Error: Data not preprocessed. Cannot train final model.")
            raise ValueError("Data not preprocessed for final model training.")

        model_params_to_use = params if params else self.best_hyperparams
        if not model_params_to_use:
            print("Pipeline Warning: No best hyperparameters found or provided. Using initial defaults from config.")
            model_params_to_use = self.cfg.get('model_params', {}).get('global_xgboost', {}).copy() # Use .copy()
            model_params_to_use.pop('tuning', None) # Remove tuning sub-dict

        final_xgb_model_params = {
            'objective': self.cfg.get('model_params', {}).get('global_xgboost', {}).get('objective', 'reg:squarederror'),
            'eval_metric': self.cfg.get('model_params', {}).get('global_xgboost', {}).get('eval_metric', 'rmse'),
            'tree_method': 'hist',
            'random_state': self.cfg.get('project_setup', {}).get('random_seed', 42),
            **model_params_to_use
        }
        
        self.model = xgb.XGBRegressor(**final_xgb_model_params)
        # For final model, typically train on X_train (or X_train + X_val) without early stopping against X_val
        # unless n_estimators was part of tuning and is now fixed.
        print(f"Training final model on X_train (shape: {self.X_train.shape})")
        self.model.fit(self.X_train, self.y_train, verbose=False) 
        print("Pipeline: Final model trained.")

    def evaluate(self, data_split='test'):
        print(f"Pipeline: Evaluating model on {data_split} set...")
        if self.model is None:
            print("Pipeline Error: Model not trained. Cannot evaluate.")
            return None
        if self.scaler is None:
             print("Pipeline Error: Scaler not available. Cannot evaluate correctly.")
             return None

        X_eval, y_eval_scaled = None, None
        # --- MODIFICATION START for train/val/test evaluation ---
        if data_split == 'test' and self.X_test is not None:
            X_eval, y_eval_scaled = self.X_test, self.y_test
        elif data_split == 'validation' and self.X_val is not None:
            X_eval, y_eval_scaled = self.X_val, self.y_val
        elif data_split == 'train' and self.X_train is not None: # Added train split
            X_eval, y_eval_scaled = self.X_train, self.y_train
        # --- MODIFICATION END ---
        else:
            print(f"Pipeline Error: Data for split '{data_split}' is not available or invalid split name.")
            return None
        
        scaled_predictions = self.model.predict(X_eval)
        target_col = self.cfg['project_setup']['target_variable']

        scaled_actuals_df = pd.DataFrame(y_eval_scaled.values, columns=[target_col], index=y_eval_scaled.index)
        scaled_preds_df = pd.DataFrame(scaled_predictions, columns=[target_col], index=y_eval_scaled.index)

        inversed_predictions = inverse_transform_predictions(scaled_preds_df, target_col, self.scaler)
        inversed_actuals = inverse_transform_predictions(scaled_actuals_df, target_col, self.scaler)
        
        if inversed_predictions is not None and inversed_actuals is not None:
            rmse = mean_squared_error(inversed_actuals, inversed_predictions)
            mae = mean_absolute_error(inversed_actuals, inversed_predictions)
            r2 = r2_score(inversed_actuals, inversed_predictions)
            print(f"{data_split.capitalize()} Set Evaluation (Original Scale): RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
            return {'rmse': rmse, 'mae': mae, 'r2': r2}
        else:
            print(f"Pipeline Error: Could not inverse transform {data_split} predictions/actuals.")
            return None

    def run_full_pipeline(self, tune=True, n_trials_tuning=50):
        self.load_and_split_data()
        if self.train_df_raw is None: return "Failed at data loading/splitting."
        
        self.engineer_all_features()
        if self.train_df_featured is None: return "Failed at feature engineering."

        self.preprocess_all_data()
        if self.X_train is None: return "Failed at data preprocessing/scaling."

        if tune:
            self.tune_hyperparameters(n_trials=n_trials_tuning)
        
        self.train_final_model() 
        if self.model is None: return "Failed at final model training."

        # --- MODIFICATION START for returning all evaluations ---
        all_metrics = {}
        print("\n--- Final Model Evaluation ---")
        train_metrics = self.evaluate(data_split='train')
        if train_metrics: all_metrics['train'] = train_metrics
        
        val_metrics = self.evaluate(data_split='validation')
        if val_metrics: all_metrics['validation'] = val_metrics
            
        test_metrics = self.evaluate(data_split='test')
        if test_metrics: all_metrics['test'] = test_metrics
        
        return all_metrics