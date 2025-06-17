import torch
from torch.utils.data import Dataset

class GriddedMultitaskSeq2SeqDataset(Dataset):
    def __init__(self, x, y_dict, mask=None, temporal_only=None):
        """
        x: Tensor [N, T, C, H, W]
        y_dict: dict of tensors [N, T, 1, H, W] for each task
        mask: optional Tensor [H, W] (1=valid, 0=masked)
        temporal_only: optional Tensor [N, T, C_temp]
        """
        self.x = torch.tensor(x, dtype=torch.float32) if not torch.is_tensor(x) else x
        self.y_dict = {
            k: torch.tensor(v, dtype=torch.float32) if not torch.is_tensor(v) else v
            for k, v in y_dict.items()
        }
        self.mask = mask
        self.temporal_only = temporal_only
        self.N = self.x.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        sample = {
            "x": self.x[idx],  # [T, C, H, W]
            "y": {k: self.y_dict[k][idx] for k in self.y_dict}  # [T, 1, H, W] per task
        }
        if self.temporal_only is not None:
            sample["temporal_only"] = self.temporal_only[idx]  # [T, C_temp]
        return sample
