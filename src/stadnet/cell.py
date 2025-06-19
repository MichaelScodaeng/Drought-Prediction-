# cell.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class STADCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, bias=True, dropout_rate=0.3, max_timesteps=24):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout_rate

        padding = kernel_size // 2

        # Gate convolutions: 5 gates now (i, f_spatial, f_trend, f_event, o)
        self.conv_x = nn.Conv2d(input_channels, 5 * hidden_channels, kernel_size, padding=padding, bias=bias)
        self.conv_h = nn.Conv2d(hidden_channels, 5 * hidden_channels, kernel_size, padding=padding, bias=bias)

        # Diagonal memory link transformation (top layer memory to base)
        self.diag_link = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)

        # LayerNorm after gate combination (H, W must be fixed or dynamically set)
        self.norm_gates = nn.LayerNorm([5 * hidden_channels, 29, 16])  # Modify H, W if needed

        # Spatial attention: learn where to focus in x_t
        self.attn = nn.Sequential(
            nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()  # spatial mask A_s
        )

        # α fusion gate: learn how to mix slow vs fast temporal memory
        self.alpha_gate = nn.Sequential(
            nn.Conv2d(2 * hidden_channels + 8, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Time embedding for α gate
        self.time_embed = nn.Embedding(max_timesteps, 8)

        # Dropout
        self.dropout = nn.Dropout2d(self.dropout_rate)

    def forward(self, x_t, h_prev, c_s_prev, c_trend_prev, c_event_prev, timestep, c_top_diag=None):
        """
        x_t:      [B, C, H, W]  input at time t
        h_prev:   [B, H_c, H, W] hidden state from t-1
        c_s_prev: [B, H_c, H, W] spatial memory
        c_trend_prev: [B, H_c, H, W] slow temporal memory
        c_event_prev: [B, H_c, H, W] fast temporal memory
        timestep: int (scalar) or tensor of shape [B]
        c_top_diag: optional [B, H_c, H, W] diagonal memory from top layer t-1
        """
        B, _, H, W = x_t.shape

        # Attention map from x and h_prev
        a_input = torch.cat([x_t, h_prev], dim=1)
        A_s = self.attn(a_input)  # [B, 1, H, W]
        x_attn = x_t * A_s

        if self.dropout_rate > 0:
            x_attn = self.dropout(x_attn)

        # Gates
        gates = self.conv_x(x_attn) + self.conv_h(h_prev)
        gates = self.norm_gates(gates)
        i, f_s, f_trend, f_event, o = torch.chunk(gates, 5, dim=1)
        i = torch.sigmoid(i)
        f_s = torch.sigmoid(f_s)
        f_trend = torch.sigmoid(f_trend)
        f_event = torch.sigmoid(f_event)
        o = torch.sigmoid(o)

        # Candidate memory
        g = torch.tanh(
            self.conv_x(x_attn)[:, :self.hidden_channels] +
            self.conv_h(h_prev)[:, :self.hidden_channels]
        )

        # Diagonal memory injection if provided
        if c_top_diag is not None:
            c_event_prev = c_event_prev + self.diag_link(c_top_diag)

        # Memory updates
        c_s = f_s * c_s_prev + i * g  # spatial memory
        c_trend = f_trend * c_trend_prev + i * g  # slow temporal
        c_event = f_event * c_event_prev + i * g  # fast temporal

        # Time embedding
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep], device=x_t.device)
        t_embed = self.time_embed(timestep % self.time_embed.num_embeddings)  # [B, 8]
        t_embed = t_embed.view(B, -1, 1, 1).expand(-1, -1, H, W)  # [B, 8, H, W]

        # Fusion gate for h_trend and h_event
        alpha_input = torch.cat([c_trend, c_event, t_embed], dim=1)
        alpha = self.alpha_gate(alpha_input)  # [B, 1, H, W]

        # Dual hidden path
        h_trend = o * torch.tanh(c_trend)
        h_event = o * torch.tanh(c_event)
        h = alpha * h_trend + (1 - alpha) * h_event

        return h, c_s, c_trend, c_event
