import pandas as pd
import yaml # For loading config if we use YAML formally

# --- Configuration Loading (Conceptual) ---
# In a real script, you'd load this from config.yaml
# For now, let's define a default config dictionary for development
DEFAULT_CONFIG = {
    'data': {
        'raw_data_path': "full.csv", # Path to your full dataset
        'time_column': "time",
        'train_end_date': "2018-12-31",
        'validation_end_date': "2020-12-31"
    }
}

def load_config(config_path="../config.yaml"):
    """
    Loads configuration from a YAML file.
    If no path is provided, returns a default configuration for development.
    """
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using default config.")
            return DEFAULT_CONFIG
        except Exception as e:
            print(f"Error loading config file {config_path}: {e}. Using default config.")
            return DEFAULT_CONFIG
    else:
        print("No config path provided. Using default config for development.")
        return DEFAULT_CONFIG

# --- Data Loading and Basic Processing ---
import xarray as xr
import pandas as pd

def load_and_prepare_data(config):
    path = config["data"]["raw_data_path"]
    if path.endswith(".zarr"):
        print(f"Loading data from: {path}")
        ds = xr.open_zarr(path, consolidated=True, storage_options={"token": "anon", "asynchronous": False})

        # Safe bounds selection
        ds = ds.sel(
            time=slice("2022", "2023"),
            latitude=slice(75, 30),
                    )
        ds = ds.where((ds.longitude >= 335) | (ds.longitude <= 50), drop=True)

        # Select variables
        ds = ds[config["data"]["variables"]]

        # Convert to dataframe
        df = ds.to_dataframe().reset_index()

        # ✅ FIX: ensure time is datetime
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])

        # Drop NaNs introduced by Zarr slicing or empty areas
        df = df.dropna()
        print(f"Successfully loaded data from {path}. Shape: {df.shape}")
        return df


'''def load_and_prepare_data(config):
    """
    Loads the raw data, converts the time column to datetime,
    and sorts the data by time and location (if lat/lon exist).

    Args:
        config (dict): Configuration dictionary containing data paths and column names.

    Returns:
        pandas.DataFrame: Loaded and prepared DataFrame, or None if an error occurs.
    """
    data_path = config['data']['raw_data_path']
    time_col = config['data']['time_column']
    
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded data from {data_path}. Shape: {df.shape}")

        # Convert time column to datetime
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
            print(f"Converted column '{time_col}' to datetime.")
        else:
            print(f"Error: Time column '{time_col}' not found in the DataFrame.")
            return None

        # Sort data: crucial for time series splits and consistency
        # Assuming 'lat' and 'lon' columns exist for secondary sort key
        sort_columns = [time_col]
        if 'lat' in df.columns:
            sort_columns.append('lat')
        if 'lon' in df.columns:
            sort_columns.append('lon')
        
        df = df.sort_values(by=sort_columns).reset_index(drop=True)
        print(f"Data sorted by {sort_columns}.")
        
        return df

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None
    except Exception as e:
        print(f"An error occurred during data loading and preparation: {e}")
        return None'''

# --- Data Splitting ---

def split_data_chronologically(df, config):
    """
    Splits the DataFrame into training, validation, and test sets based on dates.

    Args:
        df (pandas.DataFrame): The input DataFrame with a datetime column.
        config (dict): Configuration dictionary with split dates and time column name.

    Returns:
        tuple: (df_train, df_val, df_test)
               Returns (None, None, None) if df is None or time column is missing.
    """
    if df is None:
        print("Error: DataFrame is None. Cannot split data.")
        return None, None, None

    time_col = config['data']['time_column']
    if time_col not in df.columns:
        print(f"Error: Time column '{time_col}' not found for splitting.")
        return None, None, None

    try:
        train_end_date = pd.to_datetime(config['data']['train_end_date'])
        val_end_date = pd.to_datetime(config['data']['validation_end_date'])
    except KeyError as e:
        print(f"Error: Missing date configuration for splitting: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error parsing split dates from config: {e}")
        return None, None, None

    print(f"Splitting data: Train ends {train_end_date}, Validation ends {val_end_date}")

    # Ensure the time column in df is datetime (should be done by load_and_prepare_data)
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        print(f"Warning: Time column '{time_col}' is not datetime. Attempting conversion.")
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except Exception as e:
            print(f"Failed to convert time column to datetime during split: {e}")
            return None, None, None
            
    df_train = df[df[time_col] <= train_end_date]
    df_val = df[(df[time_col] > train_end_date) & (df[time_col] <= val_end_date)]
    df_test = df[df[time_col] > val_end_date]

    print(f"Train set shape: {df_train.shape}, Time range: {df_train[time_col].min()} to {df_train[time_col].max() if not df_train.empty else 'N/A'}")
    print(f"Validation set shape: {df_val.shape}, Time range: {df_val[time_col].min()} to {df_val[time_col].max() if not df_val.empty else 'N/A'}")
    print(f"Test set shape: {df_test.shape}, Time range: {df_test[time_col].min()} to {df_test[time_col].max() if not df_test.empty else 'N/A'}")
    
    if df_train.empty or df_val.empty or df_test.empty:
        print("Warning: One or more data splits are empty. Check your split dates and data range.")

    return df_train, df_val, df_test


# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    print("--- Testing data_utils.py ---")
    
    # 1. Load configuration (using default for this test)
    cfg = load_config() # No path, so uses DEFAULT_CONFIG
    print("\n--- Loaded Configuration ---")
    print(cfg)

    if not cfg:
        print("Failed to load configuration. Exiting test.")
    else:
        # 2. Load and prepare data
        full_df = load_and_prepare_data(cfg)

        if full_df is not None:
            print("\nSample of loaded data (head):")
            print(full_df.head())
            print(f"\nData types:\n{full_df.dtypes}")

            # 3. Split data
            print("\n--- Splitting Data ---")
            train_df, val_df, test_df = split_data_chronologically(full_df, cfg)

            if train_df is not None and val_df is not None and test_df is not None:
                print("\nSuccessfully split data.")
                # Further checks can be added here, e.g., ensuring no overlap
                if not train_df.empty and not val_df.empty:
                    assert train_df[cfg['data']['time_column']].max() < val_df[cfg['data']['time_column']].min(), "Train/Val overlap!"
                if not val_df.empty and not test_df.empty:
                     assert val_df[cfg['data']['time_column']].max() < test_df[cfg['data']['time_column']].min(), "Val/Test overlap!"
                print("Split integrity checks passed (no obvious time overlaps).")
            else:
                print("Failed to split data.")
        else:
            print("Failed to load data. Cannot proceed with splitting test.")
    
    print("\n--- Finished testing data_utils.py ---")
