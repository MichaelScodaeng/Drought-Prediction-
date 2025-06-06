import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def create_gridded_data(df, config):
    """
    Transforms point-based spatio-temporal data (on a regular grid) into a
    gridded tensor and creates a corresponding validity mask.

    Args:
        df (pd.DataFrame): Input DataFrame sorted by time.
        config (dict): Configuration dictionary.

    Returns:
        tuple: (gridded_data, mask)
    """
    print("--- Starting Data Gridding Process (Fixed Step Method) ---")
    
    # --- 1. Get parameters from config ---
    data_cfg = config.get('data', {})
    grid_cfg = config.get('gridding', {})
    
    time_col = data_cfg.get('time_column', 'time')
    lat_col = data_cfg.get('lat_column', 'lat')
    lon_col = data_cfg.get('lon_column', 'lon')
    features_to_grid = data_cfg.get('features_to_grid', [])
    
    # Use the known, fixed step size from the config
    fixed_step = grid_cfg.get('fixed_step', 0.5)
    print(f"Using fixed grid step of: {fixed_step} degrees")
    
    # --- 2. Calculate grid dimensions based on fixed step ---
    lat_min, lat_max = df[lat_col].min(), df[lat_col].max()
    lon_min, lon_max = df[lon_col].min(), df[lon_col].max()
    
    # Calculate grid dimensions. +1 to include the last point.
    grid_h = int(round((lat_max - lat_min) / fixed_step)) + 1
    grid_w = int(round((lon_max - lon_min) / fixed_step)) + 1
    
    print(f"Grid boundaries: LAT ({lat_min:.2f}, {lat_max:.2f}), LON ({lon_min:.2f}, {lon_max:.2f})")
    print(f"Calculated grid dimensions: Height={grid_h}, Width={grid_w}")

    # --- 3. Map each (lat, lon) point to its exact grid cell (pixel) ---
    # No linear scaling needed, just direct mapping based on the fixed step.
    df['row_idx'] = ((df[lat_col] - lat_min) / fixed_step).round().astype(int)
    df['col_idx'] = ((df[lon_col] - lon_min) / fixed_step).round().astype(int)

    # --- 4. Create the validity mask from the point locations ---
    # Since we are using the native grid, no morphological closing should be needed.
    mask = np.zeros((grid_h, grid_w), dtype=int)
    valid_cells = df[['row_idx', 'col_idx']].drop_duplicates().values
    for r, c in valid_cells:
        mask[r, c] = 1
    
    n_valid_pixels = np.sum(mask)
    print(f"Created 2D validity mask ({grid_h}x{grid_w}) with {n_valid_pixels} valid data pixels.")

    # --- 5. Create the 4D data tensor ---
    time_steps = df[time_col].unique()
    n_timesteps = len(time_steps)
    n_features = len(features_to_grid)
    
    gridded_data = np.zeros((n_timesteps, grid_h, grid_w, n_features), dtype=np.float32)
    
    print(f"Pivoting data into a 4D tensor of shape ({n_timesteps}, {grid_h}, {grid_w}, {n_features})...")
    
    grouped = df.groupby([time_col, 'row_idx', 'col_idx'])[features_to_grid].mean()
    time_to_idx = {time_val: i for i, time_val in enumerate(time_steps)}

    for (time_val, r, c), values in grouped.iterrows():
        t_idx = time_to_idx[time_val]
        gridded_data[t_idx, r, c, :] = values.values

    print("--- Data Gridding Process Finished ---")
    
    return gridded_data, mask


# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    try:
        from data_utils import load_config 
        cfg = load_config("config_CNN3D.yaml")
        if not cfg.get('gridding'): raise FileNotFoundError
    except (ImportError, FileNotFoundError):
        print("Warning: Could not load config_CNN3D.yaml. Using dummy config for testing.")
        cfg = {
            'data': {'raw_data_path':"full.csv", 'time_column':'time', 'lat_column':'lat', 'lon_column':'lon', 'features_to_grid': ['spei', 'tmp']},
            'gridding': {'fixed_step': 0.5}
        }

    data_path = cfg['data']['raw_data_path']
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at '{data_path}'. Cannot run test.")
    else:
        full_df = pd.read_csv(data_path, parse_dates=[cfg['data']['time_column']])
        full_df.sort_values(by=cfg['data']['time_column'], inplace=True)

        gridded_tensor, validity_mask = create_gridded_data(full_df, cfg)

        print(f"\n--- Test Results ---")
        print(f"Shape of the final gridded data tensor: {gridded_tensor.shape}")
        print(f"Shape of the final validity mask: {validity_mask.shape}")
        
        print("\nVisualizing the final validity mask (should be solid)...")
        plt.figure(figsize=(8, 10))
        plt.imshow(validity_mask, cmap='viridis', origin='lower')
        plt.colorbar(ticks=[0, 1])
        plt.title("Final Validity Mask of Study Area (Fixed Step Method)")
        plt.xlabel("Grid Column (Longitude Index)")
        plt.ylabel("Grid Row (Latitude Index)")
        plt.show()
