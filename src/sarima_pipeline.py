import pandas as pd
import numpy as np
import yaml
import os
import json
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning) # pmdarima often gives user warnings

# Assuming these are in src/ or PYTHONPATH is set for the notebook
try:
    from src.data_utils import load_config, load_and_prepare_data
    print("SARIMA Pipeline Class: Successfully imported utility functions from data_utils.")
except ImportError as e:
    print(f"SARIMA Pipeline Class Error: Could not import from data_utils: {e}")
    # Define dummy functions if import fails
    def load_config(path=None): return {}
    def load_and_prepare_data(config=None): return None

class SarimaLocalPipeline:
    def __init__(self, config_path="config.yaml"):
        self.config_path_abs = os.path.abspath(config_path)
        print(f"SARIMA Pipeline Class: Loading config from: {self.config_path_abs}")
        self.cfg = load_config(self.config_path_abs)
        
        self.experiment_name = self.cfg.get('project_setup', {}).get('experiment_name', 'sarima_experiment')
        self.project_root_for_paths = os.path.dirname(self.config_path_abs)

        results_base_cfg = self.cfg.get('paths',{}).get('output_base_dir', 'run_outputs')
        self.run_output_dir = os.path.join(self.project_root_for_paths, results_base_cfg, self.experiment_name + "_sarima_local")
        os.makedirs(self.run_output_dir, exist_ok=True)
        print(f"SARIMA Pipeline Class: Artifacts will be saved under '{self.run_output_dir}'")
        
        self.full_raw_data = None
        self.unique_locations = []
        self.all_location_metrics = []

    def _get_abs_path_from_config_value(self, relative_path):
        if relative_path is None: return None
        if os.path.isabs(relative_path): return relative_path
        return os.path.abspath(os.path.join(self.project_root_for_paths, relative_path))

    def _load_full_data(self):
        print("SARIMA Pipeline: Loading full raw data...")
        raw_path = self.cfg.get('data', {}).get('raw_data_path')
        if not raw_path:
            print("SARIMA Pipeline Error: 'data.raw_data_path' not found in config.")
            return False
        abs_path = self._get_abs_path_from_config_value(raw_path)
        if not abs_path or not os.path.exists(abs_path):
            print(f"SARIMA Pipeline Error: Data file not found at: {abs_path}")
            return False

        temp_cfg = {'data': {'raw_data_path': abs_path, 'time_column': self.cfg['data']['time_column']}}
        self.full_raw_data = load_and_prepare_data(temp_cfg)
        if self.full_raw_data is None:
            print("SARIMA Pipeline Error: Failed to load full_raw_data."); return False
        
        lat_col, lon_col = self.cfg['data']['lat_column'], self.cfg['data']['lon_column']
        self.unique_locations = self.full_raw_data[[lat_col, lon_col]].drop_duplicates().values.tolist()
        print(f"SARIMA Pipeline: Found {len(self.unique_locations)} unique locations.")
        return True
    
    def _stationarity_test(self, timeseries_data, significance_level=0.05):
        """Performs Augmented Dickey-Fuller test for stationarity."""
        result = adfuller(timeseries_data.dropna())
        p_value = result[1]
        is_stationary = p_value < significance_level
        return is_stationary, p_value
    
    def _run_auto_arima(self, train_data, seasonal_period, exogenous_data=None):
        """Uses pmdarima.auto_arima to find the best SARIMA/SARIMAX model."""
        model = pm.auto_arima(
            y=train_data,
            X=exogenous_data,
            start_p=1, start_q=1,
            test='adf',       
            max_p=3, max_q=3, 
            m=seasonal_period,      
            d=None,           
            seasonal=True,    
            start_P=0, 
            D=1,              
            trace=False,       
            error_action='ignore',  
            suppress_warnings=True, 
            stepwise=True,
            n_jobs=-1,  # Use all available cores
            maxiter = 10 # Limit iterations for faster debugging
        )
        return model

    def _calculate_metrics(self, actuals, predictions):
        """Helper function to calculate a dictionary of metrics."""
        actuals_np = np.asarray(actuals)
        predictions_np = np.asarray(predictions)
        finite_mask = np.isfinite(actuals_np) & np.isfinite(predictions_np)
        if np.sum(finite_mask) == 0:
            return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
        rmse = mean_squared_error(actuals_np[finite_mask], predictions_np[finite_mask])
        mae = mean_absolute_error(actuals_np[finite_mask], predictions_np[finite_mask])
        r2 = r2_score(actuals_np[finite_mask], predictions_np[finite_mask])
        return {'rmse': rmse, 'mae': mae, 'r2': r2}

    def _process_location(self, location_coords, location_data):
        lat, lon = location_coords
        loc_identifier = f"lat{lat}_lon{lon}"
        print(f"\n--- Processing Location: {loc_identifier} ---")

        target_col = self.cfg['project_setup']['target_variable']
        time_col = self.cfg['data']['time_column']
        
        location_ts = location_data.set_index(time_col)[target_col]
        location_ts.index.freq = pd.infer_freq(location_ts.index)
        if location_ts.index.freq is None:
            print(f"  Warning: Could not infer frequency for {loc_identifier}.")
        else:
            print(f"  Inferred frequency: {location_ts.index.freq}")

        if len(location_ts.dropna()) < self.cfg.get('sarima_params',{}).get('min_data_points', 36):
            print(f"Warning: Insufficient data for {loc_identifier}. Skipping.")
            return None

        train_end_date_str = self.cfg['data']['train_end_date']
        val_end_date_str = self.cfg['data']['validation_end_date']
        
        train_data = location_ts[:train_end_date_str]
        val_data = location_ts[pd.to_datetime(train_end_date_str) + pd.DateOffset(days=1) : val_end_date_str]
        test_data = location_ts[pd.to_datetime(val_end_date_str) + pd.DateOffset(days=1):]

        if len(train_data.dropna()) < 24 or len(val_data.dropna()) == 0 or len(test_data.dropna()) == 0:
            print(f"Warning: Train, validation, or test set empty/too small. Skipping.")
            return None
        
        is_stationary, p_value = self._stationarity_test(train_data)
        print(f"  Stationarity (p-value): {p_value:.4f} -> {'Stationary' if is_stationary else 'Not Stationary'}")

        seasonal_period = self.cfg.get('sarima_params', {}).get('seasonal_period', 12)
        print(f"  Running auto_arima with seasonal period m={seasonal_period}...")
        try:
            sarima_model = self._run_auto_arima(train_data, seasonal_period)
            print(f"  Best model found for {loc_identifier}: {sarima_model.order} {sarima_model.seasonal_order}")
        except Exception as e:
            print(f"  Error during auto_arima for {loc_identifier}: {e}"); return None
        
        full_train_val_data = pd.concat([train_data, val_data])
        sarima_model.fit(full_train_val_data)

        location_metrics_summary = {'location': loc_identifier, 'lat': lat, 'lon': lon}
        
        # --- GENERATE FULL PREDICTIONS & EVALUATE ON ALL SPLITS ---
        # 1. Get in-sample predictions for the part the model was fitted on
        in_sample_predictions = sarima_model.predict_in_sample()
        # 2. Get out-of-sample forecasts for the test set
        test_predictions_series = sarima_model.predict(n_periods=len(test_data))
        
        # --- FIX START: Manually align test prediction index ---
        # Create a new Series for test predictions that uses the known, correct index from test_data
        test_predictions_aligned = pd.Series(test_predictions_series.values, index=test_data.index)
        # --- FIX END ---
        
        # 3. Combine them into a single series aligned with the original full time series
        full_predictions = pd.Series(index=location_ts.index, dtype=float)
        full_predictions.update(in_sample_predictions)
        full_predictions.update(test_predictions_aligned) # Use the aligned series
        
        # 4. Evaluate on each split
        train_actuals = train_data.dropna()
        if not train_actuals.empty:
            train_preds_aligned = full_predictions.loc[train_actuals.index]
            train_metrics = self._calculate_metrics(train_actuals, train_preds_aligned)
            location_metrics_summary['train'] = train_metrics
            print(f"  {loc_identifier} - Train Set:      RMSE={train_metrics['rmse']:.4f}, MAE={train_metrics['mae']:.4f}, R2={train_metrics['r2']:.4f}")

        val_actuals = val_data.dropna()
        if not val_actuals.empty:
            val_preds_aligned = full_predictions.loc[val_actuals.index]
            val_metrics = self._calculate_metrics(val_actuals, val_preds_aligned)
            location_metrics_summary['validation'] = val_metrics
            print(f"  {loc_identifier} - Validation Set: RMSE={val_metrics['rmse']:.4f}, MAE={val_metrics['mae']:.4f}, R2={val_metrics['r2']:.4f}")

        test_actuals = test_data.dropna()
        if not test_actuals.empty:
            test_preds_aligned = full_predictions.loc[test_actuals.index]
            test_metrics = self._calculate_metrics(test_actuals, test_preds_aligned)
            test_metrics.update({'order': list(sarima_model.order), 'seasonal_order': list(sarima_model.seasonal_order)})
            location_metrics_summary['test'] = test_metrics
            print(f"  {loc_identifier} - Test Set:       RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}, R2={test_metrics['r2']:.4f}")

        # 5. Pass full results to save function
        full_results_df = pd.DataFrame({'actual_spei': location_ts, 'predicted_spei': full_predictions})
        self._save_location_results(loc_identifier, full_results_df, train_end_date_str, val_end_date_str)

        return location_metrics_summary
    
    def _save_location_results(self, loc_identifier, full_results_df, train_end_date, val_end_date):
        """Saves a plot and a CSV of the results for a single location's full time series."""
        # Save CSV
        csv_filename = self.cfg.get('results',{}).get('per_location_full_pred_filename_suffix', '_full_pred.csv')
        csv_path = os.path.join(self.run_output_dir, f"{loc_identifier}{csv_filename}")
        full_results_df.to_csv(csv_path)

        # Save Plot
        plt.figure(figsize=(15, 6))
        plt.plot(full_results_df.index, full_results_df['actual_spei'], label='Actual SPEI', color='blue', alpha=0.8)
        plt.plot(full_results_df.index, full_results_df['predicted_spei'], label='Predicted SPEI (In-sample & Forecast)', color='orange', linestyle='--')
        
        # Add vertical lines to show splits
        plt.axvline(pd.to_datetime(train_end_date), color='green', linestyle=':', lw=2, label='Train/Val Split')
        plt.axvline(pd.to_datetime(val_end_date), color='red', linestyle=':', lw=2, label='Val/Test Split')

        plt.title(f'SARIMA Full Forecast vs Actuals for {loc_identifier}')
        plt.legend()
        plt.grid(True)
        plot_filename = self.cfg.get('results',{}).get('per_location_full_plot_filename_suffix', '_full_plot.png')
        plot_path = os.path.join(self.run_output_dir, f"{loc_identifier}{plot_filename}")
        plt.savefig(plot_path)
        plt.close()

    def run_pipeline(self):
        print(f"\n--- Starting SARIMA Local Pipeline Run: Experiment '{self.experiment_name}_sarima_local' ---")
        if not self._load_full_data():
            print("Pipeline Halted: Failed at data loading."); return "Failed: Data Load"
        
        self.all_location_metrics = []
        lat_col, lon_col = self.cfg['data']['lat_column'], self.cfg['data']['lon_column']

        for loc_coords in self.unique_locations:
            lat, lon = loc_coords
            location_data_df = self.full_raw_data[
                (self.full_raw_data[lat_col] == lat) &
                (self.full_raw_data[lon_col] == lon)
            ].copy()
            if location_data_df.empty: continue
            
            metrics = self._process_location(loc_coords, location_data_df)
            if metrics:
                self.all_location_metrics.append(metrics)
        
        self._save_aggregated_metrics()
        print(f"--- SARIMA Local Pipeline Run Finished ---")
        return self.all_location_metrics

    def _save_aggregated_metrics(self):
        if not self.all_location_metrics:
            print("No metrics collected from SARIMA models to save."); return
        metrics_filename = self.cfg.get('results',{}).get('metrics_filename', 'sarima_local_metrics.json')
        metrics_save_path = os.path.join(self.run_output_dir, metrics_filename)
        try:
            with open(metrics_save_path, 'w') as f: json.dump(self.all_location_metrics, f, indent=4)
            print(f"SARIMA Pipeline: Aggregated metrics saved to {metrics_save_path}")
        except Exception as e: print(f"SARIMA Pipeline Error: Could not save aggregated metrics: {e}")
