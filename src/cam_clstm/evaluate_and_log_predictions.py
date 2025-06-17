
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_and_log_predictions(model, dataloader, device, output_csv_path="predictions.csv"):
    model.eval()
    rows = []

    with torch.no_grad():
        for x, y_dict, meta in dataloader:
            x = x.to(device)
            y_dict = {k: v.to(device) for k, v in y_dict.items()}
            preds = model(x)

            for i in range(x.size(0)):  # batch dimension
                timestamp = meta['time'][i]
                lats = meta['lat']
                lons = meta['lon']
                for h in range(len(lats)):
                    for w in range(len(lons)):
                        lat = lats[h].item()
                        lon = lons[w].item()
                        for task in ['pre', 'pet', 'spei']:
                            pred_val = preds[task][i, 0, h, w].item()
                            true_val = y_dict[task][i, 0, h, w].item()
                            rows.append({
                                "time": str(timestamp),
                                "lat": lat,
                                "lon": lon,
                                "task": task,
                                "target": true_val,
                                "predicted": pred_val
                            })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved predictions to {output_csv_path}")

    # Optional: compute metrics
    metrics = {}
    for task in ['pre', 'pet', 'spei']:
        df_task = df[df["task"] == task]
        y_true = df_task["target"].values
        y_pred = df_task["predicted"].values
        metrics[task] = {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }

    return metrics
