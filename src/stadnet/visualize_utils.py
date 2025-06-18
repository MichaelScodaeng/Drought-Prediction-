import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_prediction_map(pred, title="Prediction", cmap="viridis"):
    """
    pred: [H, W] numpy or torch tensor
    """
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    plt.figure(figsize=(6, 5))
    plt.imshow(pred, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()

def plot_comparison(pred, target, mask=None, title="Prediction vs Target"):
    """
    pred, target: [H, W] tensors or arrays
    mask: [H, W] binary mask
    """
    if torch.is_tensor(pred): pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target): target = target.detach().cpu().numpy()
    if mask is not None:
        if torch.is_tensor(mask): mask = mask.cpu().numpy()
        pred = np.where(mask, pred, np.nan)
        target = np.where(mask, target, np.nan)

    diff = pred - target

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].imshow(pred, cmap="viridis")
    axs[0].set_title("Prediction")
    axs[1].imshow(target, cmap="viridis")
    axs[1].set_title("Ground Truth")
    axs[2].imshow(diff, cmap="bwr")
    axs[2].set_title("Error (Pred - GT)")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def visualize_attention_map(attn_map, title="Spatial Attention Map"):
    """
    attn_map: [1, H, W] or [H, W] tensor
    """
    if torch.is_tensor(attn_map):
        attn_map = attn_map.squeeze().detach().cpu().numpy()
    plt.figure(figsize=(5, 4))
    plt.imshow(attn_map, cmap="plasma")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
