import torch
import math

def get_temporal_positional_encoding(T, device):
    """
    Returns a tensor of shape [T, 2] with sin/cos positional encodings
    for each timestep (e.g., 12 steps)
    """
    pe = torch.zeros(T, 2)
    for t in range(T):
        angle = 2 * math.pi * t / T
        pe[t, 0] = math.sin(angle)
        pe[t, 1] = math.cos(angle)
    return pe.to(device)  # [T, 2]

def add_temporal_pe(x_seq):
    """
    Add temporal positional encoding to each timestep in input sequence.
    x_seq: [B, T, C, H, W]
    Returns: [B, T, C+2, H, W]
    """
    B, T, C, H, W = x_seq.shape
    pe = get_temporal_positional_encoding(T, x_seq.device)  # [T, 2]
    pe = pe.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)         # [1, T, 2, 1, 1]
    pe = pe.expand(B, -1, -1, H, W)                          # [B, T, 2, H, W]
    return torch.cat([x_seq, pe], dim=2)                    # [B, T, C+2, H, W]

def get_spatial_positional_encoding(H, W, device):
    """
    Returns a tensor of shape [4, H, W] with sin/cos encodings over rows/cols.
    """
    row_pos = torch.arange(H, dtype=torch.float32).unsqueeze(1).repeat(1, W) / H
    col_pos = torch.arange(W, dtype=torch.float32).unsqueeze(0).repeat(H, 1) / W
    
    row_enc = torch.stack([
        torch.sin(2 * math.pi * row_pos),
        torch.cos(2 * math.pi * row_pos)
    ], dim=0)  # [2, H, W]

    col_enc = torch.stack([
        torch.sin(2 * math.pi * col_pos),
        torch.cos(2 * math.pi * col_pos)
    ], dim=0)  # [2, H, W]

    return torch.cat([row_enc, col_enc], dim=0).to(device)  # [4, H, W]

def add_spatial_pe(x_seq):
    """
    Add spatial positional encoding to each timestep in input sequence.
    x_seq: [B, T, C, H, W]
    Returns: [B, T, C+4, H, W]
    """
    B, T, C, H, W = x_seq.shape
    pe = get_spatial_positional_encoding(H, W, x_seq.device)  # [4, H, W]
    pe = pe.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)  # [B, T, 4, H, W]
    return torch.cat([x_seq, pe], dim=2)  # [B, T, C+4, H, W]
