
import numpy as np
import pandas as pd
from src.grid_utils import create_gridded_data

def convert_csv_to_lstm_dataset(df, config, seq_len=12, target_key='spei'):
    """
    Converts raw CSV climate data into ConvLSTM-ready X and y dictionaries.

    Args:
        csv_path (str): Path to raw .csv data.
        config (dict): Configuration dictionary.
        seq_len (int): Length of input sequence.
        target_key (str): The key in the features to be used as target.

    Returns:
        tuple: X [N, T, C, H, W], y_dict {'task': [N, 1, H, W]}
    """
    gridded_data, mask = create_gridded_data(df, config)  # [T, H, W, C]

    X_all = gridded_data[..., :-1]  # [T, H, W, Cx]
    Y_all = gridded_data[..., -1:]  # [T, H, W, 1] (target is last feature)

    # Sliding windows
    X_seq = []
    Y_seq = []
    T, H, W, Cx = X_all.shape
    for t in range(T - seq_len):
        x_win = X_all[t:t+seq_len].transpose(0, 3, 1, 2)  # [T, C, H, W]
        y_win = Y_all[t+seq_len].transpose(2, 0, 1)       # [1, H, W]
        X_seq.append(x_win)
        Y_seq.append(y_win)

    X_np = np.stack(X_seq)  # [N, T, C, H, W]
    Y_np = np.stack(Y_seq)  # [N, 1, H, W]

    print(f"Created sequence data: X={X_np.shape}, y={Y_np.shape}")
    return X_np, {target_key: Y_np}, mask
