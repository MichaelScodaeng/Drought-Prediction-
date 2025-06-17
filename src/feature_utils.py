import pandas as pd
import numpy as np
import yaml
import os

# For the __main__ example usage.
try:
    from data_utils import load_config as load_data_config
    from data_utils import load_and_prepare_data, split_data_chronologically
    DATA_UTILS_AVAILABLE = True
except ImportError:
    DATA_UTILS_AVAILABLE = False
    # Dummy functions for parsing
    def load_data_config(path=None): return {}
    def load_and_prepare_data(config=None): return None
    def split_data_chronologically(df=None, config=None): return None, None, None

DEFAULT_FEATURE_CONFIG = { # Condensed for brevity
    'project_setup': {'target_variable': "spei"},
    'data': {'predictor_columns': ['tmp', 'pre'], 'time_column': "time", 'raw_data_path': "full.csv", 'train_end_date': "2018-12-31", 'validation_end_date': "2020-12-31"},
    'feature_engineering': {'columns_to_lag': [], 'lag_periods': [1, 12], 'date_features_to_extract': ['month', 'year']}
}

def load_feature_config(config_path="config.yaml"): # Simplified for brevity
    # In a real scenario, this would merge with defaults properly
    if config_path and os.path.exists(os.path.abspath(config_path)):
        with open(os.path.abspath(config_path), 'r') as f: config = yaml.safe_load(f)
    else: config = DEFAULT_FEATURE_CONFIG.copy()
    if not config.get('feature_engineering', {}).get('columns_to_lag'):
        target_var = config.get('project_setup', {}).get('target_variable')
        predictors = config.get('data', {}).get('predictor_columns')
        if target_var and predictors: config['feature_engineering']['columns_to_lag'] = [target_var] + predictors
    return config

def create_lagged_features(df, group_by_cols, columns_to_lag, lag_periods, is_debug_location=False):
    base_df = df.copy().reset_index(drop=True)
    lagged_feature_list = []

    if is_debug_location:
        print(f"  DEBUG (create_lagged_features): Input df shape {df.shape}, head:\n{df.head(3)}")

    for col_to_lag in columns_to_lag:
        if col_to_lag not in base_df.columns:
            continue
        for lag in lag_periods:
            new_col_name = f"{col_to_lag}_lag_{lag}"
            if group_by_cols:
                lagged_col = base_df.groupby(group_by_cols, group_keys=False)[col_to_lag].shift(lag)
            else:
                lagged_col = base_df[col_to_lag].shift(lag)
            lagged_feature_list.append(lagged_col.rename(new_col_name))

            if is_debug_location and lag == max(lag_periods):
                print(f"    DEBUG: Sample of lagged col {new_col_name}:\n{lagged_col.head(15)}")

    # Concatenate all lagged columns at once (avoids fragmentation)
    if lagged_feature_list:
        lagged_df = pd.concat(lagged_feature_list, axis=1)
        base_df = pd.concat([base_df, lagged_df], axis=1)

    return base_df


def create_date_features(df, time_column_name, date_features_to_extract): # Simplified
    df_with_date_features = df.copy()
    if time_column_name not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[time_column_name]): return df_with_date_features
    for feature in date_features_to_extract:
        if feature == 'month': df_with_date_features['month'] = df_with_date_features[time_column_name].dt.month
        elif feature == 'year': df_with_date_features['year'] = df_with_date_features[time_column_name].dt.year
    return df_with_date_features

# MODIFIED engineer_features with IS_DEBUG_LOCATION
def engineer_features(df, config, is_debug_location=False): # Added debug flag
    if df is None or df.empty:
        print("Input DataFrame to engineer_features is None or empty. Returning as is.")
        return df
    
    if is_debug_location:
        print(f"  DEBUG (engineer_features): START for debug location. Input df head:\n{df.head(15)}")

    df_engineered = df.copy()
    group_by_cols = []
    if 'lat' in df_engineered.columns: group_by_cols.append('lat')
    if 'lon' in df_engineered.columns: group_by_cols.append('lon')

    feature_eng_config = config.get('feature_engineering', {})
    cols_to_lag = feature_eng_config.get('columns_to_lag', [])
    lag_periods = feature_eng_config.get('lag_periods', [])
    if not cols_to_lag: 
        target_var = config.get('project_setup', {}).get('target_variable')
        predictors = config.get('data', {}).get('predictor_columns')
        if target_var and predictors:
            cols_to_lag = [target_var] + predictors
    
    if is_debug_location:
        print(f"  DEBUG: Columns to lag: {cols_to_lag}")
        print(f"  DEBUG: Lag periods: {lag_periods}")

    if cols_to_lag and lag_periods:
        df_engineered = create_lagged_features(df_engineered, group_by_cols, cols_to_lag, lag_periods, is_debug_location)
        if is_debug_location:
            print(f"  DEBUG: After create_lagged_features for debug loc (head of first 15 rows, relevant columns):")
            # Select time, original spei, and one of the longest lag columns for spei
            longest_spei_lag_col = f"{config.get('project_setup',{}).get('target_variable')}_lag_{max(lag_periods)}"
            debug_cols_after_lag = [config.get('data',{}).get('time_column')] + [config.get('project_setup',{}).get('target_variable')]
            if longest_spei_lag_col in df_engineered.columns:
                 debug_cols_after_lag.append(longest_spei_lag_col)
            else: # If longest lag for target not found, pick one from a predictor
                first_pred_for_lag = config.get('data',{}).get('predictor_columns')[0]
                longest_pred_lag_col = f"{first_pred_for_lag}_lag_{max(lag_periods)}"
                if longest_pred_lag_col in df_engineered.columns:
                    debug_cols_after_lag.append(longest_pred_lag_col)

            print(df_engineered[debug_cols_after_lag].head(15))


    time_col = config.get('data', {}).get('time_column', 'time')
    date_features = feature_eng_config.get('date_features_to_extract', []) 
    if date_features:
        df_engineered = create_date_features(df_engineered, time_col, date_features)
        if is_debug_location:
            print(f"  DEBUG: After create_date_features for debug loc (head):\n{df_engineered.head(15)}")
        
    initial_rows = len(df_engineered)
    if cols_to_lag and lag_periods: 
        if is_debug_location:
            print(f"  DEBUG: BEFORE dropna for debug loc. Shape: {df_engineered.shape}")
            # For the first 15 rows of the debug location, check if any are NaN
            print(f"  DEBUG: NaNs in any column for first 15 rows (debug loc):\n{df_engineered.head(15).isna().any(axis=1)}")
            # Specifically check the longest lag columns for the first 15 rows
            for col_to_inspect in cols_to_lag:
                longest_lag_col_name = f"{col_to_inspect}_lag_{max(lag_periods)}"
                if longest_lag_col_name in df_engineered.columns:
                    print(f"    DEBUG: NaNs in '{longest_lag_col_name}' for first 15 rows:\n{df_engineered[longest_lag_col_name].head(15).isna().values}")


        df_engineered.dropna(inplace=True) 
        final_rows = len(df_engineered)
        print(f"Dropped {initial_rows - final_rows} rows due to NaNs after feature engineering (lags).")
        if is_debug_location:
            print(f"  DEBUG: AFTER dropna for debug loc. Shape: {df_engineered.shape}")
            print(f"  DEBUG: Head of df_engineered after dropna (debug loc):\n{df_engineered.head(15)}")

    return df_engineered

if __name__ == '__main__':
    print("--- Testing feature_utils.py (v3 - Detailed Debug) ---")
    # This __main__ is simplified and won't use the IS_DEBUG_LOCATION effectively
    # unless you modify it to process a single, known problematic location.
    # The real test will be running the full XGBoostLocalPipeline.
    cfg = load_feature_config("config.yaml")
    if DATA_UTILS_AVAILABLE:
        full_df = load_and_prepare_data(cfg)
        if full_df is not None and not full_df.empty:
            # Example: Isolate one location for detailed test if needed
            # test_loc_df = full_df[(full_df['lat'] == 6.25) & (full_df['lon'] == 101.25)].copy()
            # if not test_loc_df.empty:
            #     print(f"Testing with isolated location data (shape: {test_loc_df.shape})")
            #     featured_df = engineer_features(test_loc_df, cfg, is_debug_location=True)
            #     print("Featured isolated location data sample:")
            #     print(featured_df.head())
            # else:
            #     print("Could not isolate test location data.")
            print("Main test block in feature_utils.py needs specific setup for is_debug_location. Best tested via pipeline.")
    print("\n--- Finished testing feature_utils.py (v3) ---")
