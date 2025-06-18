# stacked_model.py
import torch
import torch.nn as nn
from .cell import STADCell
class TemporalSelfAttention(nn.Module):
    def __init__(self, embed_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = embed_dim ** -0.5

        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.to_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B, T, -1)  # [B, T, C*H*W]

        qkv = self.to_qkv(x)  # [B, T, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.heads, -1).transpose(1, 2)  # [B, heads, T, D']
        k = k.view(B, T, self.heads, -1).transpose(1, 2)
        v = v.view(B, T, self.heads, -1).transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, T, T]
        attn = attn_weights.softmax(dim=-1)

        out = attn @ v  # [B, heads, T, D']
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # [B, T, D]
        out = self.to_out(out)

        return out.view(B, T, C, H, W)

class STADNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, n_layers,
                    output_targets=["PET"], kernel_size=3, dropout_rate=0.3,
                    use_temporal_attn=False):
            super().__init__()
            self.use_temporal_attn = use_temporal_attn

            self.n_layers = n_layers
            self.output_targets = output_targets
            self.hidden_channels = hidden_channels

            self.cells = nn.ModuleList([
                STADCell(input_channels if i == 0 else hidden_channels[i-1],
                                hidden_channels[i], kernel_size, dropout_rate=dropout_rate)
                for i in range(n_layers)
            ])

            self.output_head = nn.Conv2d(hidden_channels[-1], len(output_targets), kernel_size=1)

            if self.use_temporal_attn:
                self.temporal_attn = TemporalSelfAttention(embed_dim=input_channels)
    def forward(self, x_seq):
        """
        x_seq: [B, T, C, H, W] input sequence
        Returns: [B, len(output_targets), H, W]
        """
        if self.use_temporal_attn:
            x_seq = self.temporal_attn(x_seq)
        B, T, C, H, W = x_seq.shape

        h = [torch.zeros(B, hc, H, W, device=x_seq.device) for hc in self.hidden_channels]
        c_s = [torch.zeros_like(h_i) for h_i in h]
        c_trend = [torch.zeros_like(h_i) for h_i in h]
        c_event = [torch.zeros_like(h_i) for h_i in h]

        for t in range(T):
            x_t = x_seq[:, t]  # [B, C, H, W]
            prev_top_c_event = c_event[-1].clone() if self.n_layers > 1 else None

            for l in range(self.n_layers):
                cell = self.cells[l]

                if l > 0:
                    x_t = x_t + h[l-1]  # residual connection

                diag_input = prev_top_c_event if l == 0 and prev_top_c_event is not None else None
                h[l], c_s[l], c_trend[l], c_event[l] = cell(x_t, h[l], c_s[l], c_trend[l], c_event[l], t, c_top_diag=diag_input)

        y = self.output_head(h[-1])  # Final output
        return y
