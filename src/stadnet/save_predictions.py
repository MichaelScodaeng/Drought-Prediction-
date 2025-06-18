import torch
import pandas as pd
import os
from src.stadnet.pe_utils import add_temporal_pe, add_spatial_pe

def save_predictions_to_csv(model, dataloader, land_mask, output_targets, raw_df, output_dir):
    """
    Save model predictions and true values to separate CSV files per target.
    Each file will have: time, lat, lon, true_value, predicted
    """
    model.eval()
    land_mask = land_mask.bool()
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Init results containers
    target_records = {target: [] for target in output_targets}

    with torch.no_grad():
        for batch_idx, (x_seq, y_true) in enumerate(dataloader):
            x_seq = x_seq.cuda()
            y_true = y_true.cuda()

            x_seq = add_temporal_pe(x_seq)
            x_seq = add_spatial_pe(x_seq)
            y_pred = model(x_seq)  # [B, C, H, W]

            B, C, H, W = y_pred.shape
            time_idx_base = dataloader.dataset.input_steps + batch_idx

            for b in range(B):
                time_step = time_idx_base + b
                if time_step >= len(raw_df['time'].unique()):
                    continue  # safety check
                time_val = raw_df['time'].unique()[time_step]

                for i, target in enumerate(output_targets):
                    pred_map = y_pred[b, i].cpu()
                    true_map = y_true[b, i].cpu()

                    for r in range(H):
                        for c in range(W):
                            if land_mask[r, c]:
                                lat = raw_df['lat'].unique()[r]
                                lon = raw_df['lon'].unique()[c]
                                target_records[target].append({
                                    'time': time_val,
                                    'lat': lat,
                                    'lon': lon,
                                    'true_value': true_map[r, c].item(),
                                    'predicted': pred_map[r, c].item()
                                })

    # Save each target's results to CSV
    for target, records in target_records.items():
        df_out = pd.DataFrame(records)
        csv_path = os.path.join(output_dir, f"{target}_predictions.csv")
        df_out.to_csv(csv_path, index=False)
        print(f"Saved {target} predictions to: {csv_path}")
