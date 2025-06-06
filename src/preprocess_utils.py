import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler # Keep others for flexibility
import joblib # For saving and loading the scaler
import os
import yaml

# Assuming data_utils.py is in the same src directory or accessible in PYTHONPATH
# This is for the __main__ example usage.
# In a structured project, you'd import these modules properly.
try:
    from data_utils import load_config as load_data_config 
    from data_utils import load_and_prepare_data, split_data_chronologically
except ImportError:
    print("Warning: Could not import from data_utils_v1 for __main__ block. Ensure it's accessible.")
    # Define dummy functions if import fails, so __main__ can be parsed, though it won't run fully.
    def load_data_config(path=None): return {}
    def load_and_prepare_data(config=None): return None
    def split_data_chronologically(df=None, config=None): return None, None, None


# --- Default Configuration (for standalone testing or fallback) ---
DEFAULT_PREPROCESS_CONFIG = {
    'project_setup': {
        'target_variable': "spei"
    },
    'data': {
        'predictor_columns': ['tmp', 'dtr', 'cld', 'tmx', 'tmn', 'wet', 'vap', 'soi', 'dmi', 'pdo', 'nino4', 'nino34', 'nino3', 'pre', 'pet'],
        # Include data loading config here if not merging with data_utils's config for testing
        'raw_data_path': "data/full.csv",
        'time_column': "time",
        'train_end_date': "2018-12-31",
        'validation_end_date': "2020-12-31"
    },
    'scaling': {
        'method': "robust", # 'robust', 'standard', 'minmax'
        'scaler_path': "models_saved/robust_scaler.joblib" # Default path to save/load scaler
    }
}

def load_preprocess_config(config_path="config.yaml"):
    """Loads preprocessing specific config, falling back to defaults."""
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Preprocessing configuration partially loaded from {config_path}")
            # Merge with default to ensure all keys are present
            # This is a simple merge, more sophisticated merging might be needed for nested dicts
            merged_config = DEFAULT_PREPROCESS_CONFIG.copy()
            for key, value in config.items():
                if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
            return merged_config
        except Exception as e:
            print(f"Error loading {config_path}: {e}. Using default preprocess config.")
            return DEFAULT_PREPROCESS_CONFIG
    else:
        if config_path: # if path was given but not found
            print(f"Warning: Config file {config_path} not found. Using default preprocess config.")
        else: # if no path was given at all
             print("No config path provided. Using default preprocess config.")
        return DEFAULT_PREPROCESS_CONFIG

# --- Scaling Functions ---

def get_scaler(method="robust"):
    """Returns a scaler object based on the method name."""
    if method == "robust":
        return RobustScaler()
    elif method == "standard":
        return StandardScaler()
    elif method == "minmax":
        return MinMaxScaler()
    else:
        print(f"Warning: Unknown scaler method '{method}'. Defaulting to RobustScaler.")
        return RobustScaler()

def scale_data(df_train, df_val, df_test, config):
    """
    Scales the target and predictor columns in train, validation, and test sets.
    The scaler is fitted ONLY on the training data.

    Args:
        df_train (pd.DataFrame): Training data.
        df_val (pd.DataFrame): Validation data.
        df_test (pd.DataFrame): Test data.
        config (dict): Configuration dictionary.

    Returns:
        tuple: (scaled_df_train, scaled_df_val, scaled_df_test, fitted_scaler)
               Returns (None, None, None, None) if input DataFrames are invalid.
    """
    if df_train is None or df_val is None or df_test is None:
        print("Error: One or more input DataFrames for scaling is None.")
        return None, None, None, None
    
    if df_train.empty:
        print("Error: Training DataFrame is empty. Cannot fit scaler.")
        return df_train.copy(), df_val.copy(), df_test.copy(), None # Return copies to avoid modifying originals if they are passed further

    target_col = config.get('project_setup', {}).get('target_variable', 'spei')
    predictor_cols = config.get('data', {}).get('predictor_columns', [])
    scaling_method = config.get('scaling', {}).get('method', 'robust')

    columns_to_scale = [target_col] + [col for col in predictor_cols if col in df_train.columns]
    
    # Ensure all columns to scale actually exist in the training data
    actual_columns_to_scale = [col for col in columns_to_scale if col in df_train.columns]
    missing_cols = set(columns_to_scale) - set(actual_columns_to_scale)
    if missing_cols:
        print(f"Warning: The following columns intended for scaling were not found in df_train and will be skipped: {missing_cols}")
    
    if not actual_columns_to_scale:
        print("Error: No valid columns found to scale in df_train.")
        return df_train.copy(), df_val.copy(), df_test.copy(), None

    print(f"Columns to be scaled using {scaling_method} scaler: {actual_columns_to_scale}")

    scaler = get_scaler(scaling_method)

    # Create copies to avoid SettingWithCopyWarning and modify copies
    scaled_df_train = df_train.copy()
    scaled_df_val = df_val.copy()
    scaled_df_test = df_test.copy()

    # Fit the scaler on the training data for the specified columns
    try:
        scaler.fit(df_train[actual_columns_to_scale])
    except Exception as e:
        print(f"Error fitting scaler on training data for columns {actual_columns_to_scale}: {e}")
        return df_train.copy(), df_val.copy(), df_test.copy(), None


    # Transform train, val, and test sets
    # Important: Only transform columns that exist in each respective dataframe
    for df_copy in [scaled_df_train, scaled_df_val, scaled_df_test]:
        cols_in_df = [col for col in actual_columns_to_scale if col in df_copy.columns]
        if cols_in_df:
            try:
                df_copy[cols_in_df] = scaler.transform(df_copy[cols_in_df])
            except Exception as e:
                print(f"Error transforming columns {cols_in_df} in a DataFrame: {e}")
                # Decide how to handle: skip this df's scaling or return error
        else:
            print(f"Warning: No columns to scale found in one of the dataframes during transform step.")


    print("Data scaling complete.")
    return scaled_df_train, scaled_df_val, scaled_df_test, scaler

def save_scaler(scaler, file_path):
    """Saves the fitted scaler to a file using joblib."""
    if scaler is None:
        print("Error: Scaler is None. Nothing to save.")
        return
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(scaler, file_path)
        print(f"Scaler saved to {file_path}")
    except Exception as e:
        print(f"Error saving scaler to {file_path}: {e}")

def load_scaler(file_path):
    """Loads a scaler from a file using joblib."""
    try:
        scaler = joblib.load(file_path)
        print(f"Scaler loaded from {file_path}")
        return scaler
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading scaler from {file_path}: {e}")
        return None

def inverse_transform_predictions(scaled_predictions_df, target_column_name, scaler):
    """
    Inverse transforms predictions for the target variable.
    Assumes predictions_df is a DataFrame with the target_column scaled.

    Args:
        scaled_predictions_df (pd.DataFrame): DataFrame containing scaled predictions
                                             for the target variable and potentially other scaled columns
                                             used during scaler fitting.
        target_column_name (str): The name of the target variable column.
        scaler (sklearn.base.TransformerMixin): The fitted scaler object that was used
                                                for the original scaling.

    Returns:
        pd.Series: A Series with the inverse-transformed target variable predictions.
                   Returns None if an error occurs.
    """
    if scaler is None:
        print("Error: Scaler is None. Cannot inverse transform.")
        return None
    if target_column_name not in scaled_predictions_df.columns:
        print(f"Error: Target column '{target_column_name}' not found in predictions DataFrame.")
        return None
    
    if not hasattr(scaler, 'feature_names_in_') or scaler.feature_names_in_ is None:
        print("Error: Scaler was not fitted with feature names. Cannot reliably inverse transform a single column.")
        print("Please ensure the scaler was fitted on a DataFrame with column names.")
        # Attempt to inverse transform assuming the single column structure if scaler allows
        if scaler.n_features_in_ == 1: # Heuristic for single feature scaler
             try:
                inversed_preds = scaler.inverse_transform(scaled_predictions_df[[target_column_name]])
                return pd.Series(inversed_preds.flatten(), name=target_column_name + "_inversed", index=scaled_predictions_df.index)
             except Exception as e:
                print(f"Error during fallback inverse transform: {e}")
                return None
        return None


    try:
        # Create a DataFrame with the same columns the scaler was fitted on.
        # Fill with dummy values (e.g., 0) for non-target columns,
        # and put the scaled target predictions in the correct column.
        num_features_fitted = len(scaler.feature_names_in_)
        temp_df_for_inverse = pd.DataFrame(0, index=scaled_predictions_df.index, columns=scaler.feature_names_in_)
        
        if target_column_name not in temp_df_for_inverse.columns:
            print(f"Error: Target column '{target_column_name}' was not among features scaler was fitted on: {scaler.feature_names_in_}")
            return None

        temp_df_for_inverse[target_column_name] = scaled_predictions_df[target_column_name]

        # Perform inverse transformation
        inversed_full_df = scaler.inverse_transform(temp_df_for_inverse)
        
        # Extract the inverse-transformed target column
        # Find the index of the target column in the scaler's feature names
        target_col_idx = list(scaler.feature_names_in_).index(target_column_name)
        inversed_target_predictions = inversed_full_df[:, target_col_idx]

        return pd.Series(inversed_target_predictions, name=target_column_name + "_inversed", index=scaled_predictions_df.index)

    except Exception as e:
        print(f"Error during inverse transformation: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    print("--- Testing preprocess_utils.py ---")
    
    # Load configuration (ideally, this would be a unified config)
    # For this test, we rely on data_utils to load its part and merge manually for preprocess_utils
    # A better approach would be a single config file parsed once.
    
    # Load data config first using data_utils loader
    # This assumes config.yaml is in the project root
    # For testing, it's often easier if data_utils.py itself can run with a default path
    # or we explicitly provide the path if needed.
    # If 'data_utils_v1' is not found, dummy functions are used, so this part will be skipped.
    if 'load_and_prepare_data' in globals() and callable(globals()['load_and_prepare_data']):
        print("Attempting to load data using functions from data_utils_v1...")
        data_cfg_path = "config.yaml"  # Assuming it's in the CWD when running this script
        
        # If config.yaml exists and data_utils.py can parse it for its needs:
        # data_cfg = load_data_config(data_cfg_path)
        # For simplicity in this isolated test, we'll use the DEFAULT_PREPROCESS_CONFIG
        # that includes data paths.
        cfg = DEFAULT_PREPROCESS_CONFIG
        print(f"Using config for preprocess_utils test. Data path: {cfg['data']['raw_data_path']}")

        full_df = load_and_prepare_data(cfg) # cfg contains data path for load_and_prepare_data

        if full_df is not None:
            train_df, val_df, test_df = split_data_chronologically(full_df, cfg)

            if train_df is not None and not train_df.empty:
                # Scale data
                scaled_train, scaled_val, scaled_test, fitted_sclr = scale_data(train_df, val_df, test_df, cfg)

                if fitted_sclr:
                    print("\nData scaled successfully.")
                    print("Scaled train head:\n", scaled_train.head())
                    
                    # Save the scaler
                    scaler_file_path = cfg['scaling']['scaler_path']
                    save_scaler(fitted_sclr, scaler_file_path)

                    # Load the scaler back
                    loaded_sclr = load_scaler(scaler_file_path)

                    if loaded_sclr:
                        print("Scaler loaded successfully from file.")
                        # Example: Inverse transform SPEI predictions from the validation set
                        # Create a dummy scaled predictions DataFrame for testing inverse_transform
                        # It should contain the target column and potentially other columns the scaler was fitted on
                        
                        target_col = cfg['project_setup']['target_variable']
                        
                        if not scaled_val.empty and target_col in scaled_val.columns:
                            # Let's make dummy predictions: just take the scaled target from validation
                            dummy_preds_df = pd.DataFrame(scaled_val[[target_col]]) 
                            
                            print(f"\nInverse transforming dummy '{target_col}' predictions (from scaled_val)...")
                            inversed_preds = inverse_transform_predictions(dummy_preds_df, target_col, loaded_sclr)
                            
                            if inversed_preds is not None:
                                print(f"Sample of inverse-transformed '{target_col}' predictions:\n", inversed_preds.head())
                                # Compare with original validation data's target column
                                original_val_target = val_df[target_col].reset_index(drop=True)
                                inversed_preds_reset = inversed_preds.reset_index(drop=True)
                                comparison = pd.DataFrame({
                                    'original': original_val_target.head(),
                                    'inversed_prediction': inversed_preds_reset.head()
                                })
                                print("\nComparison (Original vs Inversed Dummy Prediction):")
                                print(comparison)
                                # Check if they are close (should be, as we used actual scaled values)
                                if pd.Series(abs(comparison['original'] - comparison['inversed_prediction']) < 1e-9).all():
                                    print("Inverse transform seems consistent with original values for this test.")
                                else:
                                    print("Discrepancy found in inverse transform test.")
                            else:
                                print("Failed to inverse transform predictions.")
                        else:
                            print(f"Validation set is empty or target column '{target_col}' missing, skipping inverse transform test.")
                else:
                    print("Scaling failed.")
            else:
                print("Training data is empty or None after split. Cannot test scaling.")
        else:
            print("Failed to load data (full_df is None). Cannot test preprocessing.")
    else:
        print("data_utils functions not available. Skipping full integration test in __main__.")

    print("\n--- Finished testing preprocess_utils.py ---")

