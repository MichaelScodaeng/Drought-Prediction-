import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from src.adm_prednet.masked_loss import masked_mse


def evaluate_model(model, dataloader, land_mask, output_targets):
    """
    Evaluate model on provided dataloader.
    Computes RMSE, MAE, R^2 for each target variable.
    """
    model.eval()
    all_preds = {target: [] for target in output_targets}
    all_truths = {target: [] for target in output_targets}
    land_mask = land_mask.bool()

    with torch.no_grad():
        for x_seq, y_true in dataloader:
            x_seq = x_seq.cuda()
            y_true = y_true.cuda()

            # Apply PE if used during training
            from src.adm_prednet.pe_utils import add_temporal_pe, add_spatial_pe
            x_seq = add_temporal_pe(x_seq)
            x_seq = add_spatial_pe(x_seq)

            y_pred = model(x_seq)  # [B, C, H, W]

            for i, target in enumerate(output_targets):
                pred_i = y_pred[:, i]  # [B, H, W]
                true_i = y_true[:, i]  # [B, H, W]

                pred_flat = pred_i[:, land_mask].detach().cpu().numpy().flatten()
                true_flat = true_i[:, land_mask].detach().cpu().numpy().flatten()

                all_preds[target].extend(pred_flat)
                all_truths[target].extend(true_flat)

    # Compute metrics
    metrics = {}
    for target in output_targets:
        y_true = np.array(all_truths[target])
        y_pred = np.array(all_preds[target])

        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics[target] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

    return metrics
