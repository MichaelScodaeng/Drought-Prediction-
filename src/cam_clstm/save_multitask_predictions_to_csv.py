
import os
import pandas as pd
import numpy as np

def save_multitask_predictions_to_csv(predictions, times, latitudes, longitudes, output_path):
    """
    Save CAM-ConvLSTM predictions to CSV format.

    Parameters:
        predictions: list of dicts with 'task' and 'pred' keys. Each pred is [H, W] array.
        times: list of timestamps corresponding to predictions (length = len(predictions) // 3)
        latitudes: 1D array of latitudes (H,)
        longitudes: 1D array of longitudes (W,)
        output_path: path to save CSV file
    """
    rows = []

    time_index = 0
    num_tasks = len(set(p['task'] for p in predictions))

    for i, pred_dict in enumerate(predictions):
        task = pred_dict["task"]
        pred_grid = pred_dict["pred"]  # shape: [H, W]

        # Infer time index (assuming grouped by time)
        if i % num_tasks == 0 and i > 0:
            time_index += 1

        time_str = str(times[time_index]) if times else f"time_{time_index}"
        for i_lat, lat in enumerate(latitudes):
            for i_lon, lon in enumerate(longitudes):
                rows.append({
                    "time": time_str,
                    "lat": lat,
                    "lon": lon,
                    "task": task,
                    "predicted": pred_grid[i_lat, i_lon]
                })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to: {output_path}")
