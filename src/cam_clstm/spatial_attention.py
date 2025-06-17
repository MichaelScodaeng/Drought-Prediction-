# === Module: spatial_attention.py ===
import torch
import torch.nn as nn
import math

class SpatialAttention2D(nn.Module):
    """
    Learns a spatial attention map over each [C, H, W] frame.
    Multiplies original input with learned weights in [0, 1].
    """
    def __init__(self, in_channels, hidden_channels=8):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()  # output in [0, 1]
        )

    def forward(self, x):
        """
        x: Tensor of shape [B, C, H, W]
        returns: [B, C, H, W] with spatial attention applied
        """
        alpha = self.attention(x)               # [B, 1, H, W]
        return x * alpha                        # broadcasting over channels


# === Positional Encoding ===
def generate_positional_encoding_2d(height, width, channels):
    """
    Generates 2D sinusoidal positional encodings.
    Returns: [C, H, W] tensor
    """
    if channels % 4 != 0:
        raise ValueError("Channels must be divisible by 4 for 2D sinusoidal encoding")

    pe = torch.zeros(channels, height, width)
    c_half = channels // 2
    div_term = torch.exp(torch.arange(0, c_half, 2) * -(math.log(10000.0) / c_half))

    pos_w = torch.arange(0, width).unsqueeze(1)
    pos_h = torch.arange(0, height).unsqueeze(1)

    pe[0:c_half:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:c_half:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)

    pe[c_half::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[c_half+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class PositionalEncoding2D(nn.Module):
    def __init__(self, height, width, channels):
        super().__init__()
        pe = generate_positional_encoding_2d(height, width, channels)
        self.register_buffer('positional_encoding', pe)

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns: [B, C + pos, H, W]
        """
        pe = self.positional_encoding.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return torch.cat([x, pe], dim=1)


# === Enhanced ConvLSTM Input Block ===
class EnhancedConvLSTMInputBlock(nn.Module):
    """
    Combines positional encoding and spatial attention for ConvLSTM input.
    """
    def __init__(self, input_channels, height, width, pos_channels=4, use_pos_enc=True, use_spatial_attn=True):
        super().__init__()
        self.use_pos_enc = use_pos_enc
        self.use_spatial_attn = use_spatial_attn
        self.pos_channels = pos_channels if use_pos_enc else 0

        if use_pos_enc:
            self.pos_encoder = PositionalEncoding2D(height, width, pos_channels)

        if use_spatial_attn:
            self.spatial_attention = SpatialAttention2D(input_channels + self.pos_channels)

    def forward(self, x):
        """
        x: Tensor [B, T, C, H, W]
        Returns: [B, T, C', H, W]
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        if self.use_pos_enc:
            x = self.pos_encoder(x)

        if self.use_spatial_attn:
            x = self.spatial_attention(x)

        C_out = x.shape[1]
        x = x.view(B, T, C_out, H, W)
        return x


# === Integration Plan ===
# Use EnhancedConvLSTMInputBlock before ConvLSTM in your model.
# Example:
#   input_block = EnhancedConvLSTMInputBlock(C, H, W)
#   x = input_block(x)  # Apply PE + attention
#   conv_out = convlstm(x)

# Let me know when you want to wire this into your full pipeline.
