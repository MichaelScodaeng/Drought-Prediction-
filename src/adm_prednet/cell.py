# cell.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ADMConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, bias=True, dropout_rate=0.3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout_rate  # Store dropout_rate

        padding = kernel_size // 2

        # Gate convolutions
        self.conv_x = nn.Conv2d(input_channels, 4 * hidden_channels, kernel_size, padding=padding, bias=bias)
        self.conv_h = nn.Conv2d(hidden_channels, 4 * hidden_channels, kernel_size, padding=padding, bias=bias)

        # Spatial attention: learn where to focus in x_t
        self.attn = nn.Sequential(
            nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()  # spatial mask A_s
        )

        # α fusion gate: learn how to mix C_s and C_t
        self.alpha_gate = nn.Sequential(
            nn.Conv2d(2 * hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Dropout layer
        self.dropout = nn.Dropout2d(self.dropout_rate)

    def forward(self, x_t, h_prev, c_s_prev, c_t_prev):
        """
        x_t:      [B, C, H, W]  input at time t
        h_prev:   [B, H_c, H, W] hidden state from t-1
        c_s_prev: [B, H_c, H, W] spatial memory
        c_t_prev: [B, H_c, H, W] temporal memory
        """
        # Attention map from x and h_prev
        a_input = torch.cat([x_t, h_prev], dim=1)
        A_s = self.attn(a_input)        # [B, 1, H, W]
        
        x_attn = x_t * A_s

        # Apply dropout if dropout_rate > 0
        if self.dropout_rate > 0:
            x_attn = self.dropout(x_attn)  # Apply dropout on the attention-weighted input

        # Gate computation
        gates = self.conv_x(x_attn) + self.conv_h(h_prev)
        i, f_s, f_t, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f_s = torch.sigmoid(f_s)
        f_t = torch.sigmoid(f_t)
        o = torch.sigmoid(o)

        # Candidate memory
        g = torch.tanh(
            self.conv_x(x_attn)[:, :self.hidden_channels] +
            self.conv_h(h_prev)[:, :self.hidden_channels]
        )

        # Memory updates
        c_s = f_s * c_s_prev + i * g
        c_t = f_t * c_t_prev + i * g

        # Fusion gate
        alpha_input = torch.cat([c_s, c_t], dim=1)
        alpha = self.alpha_gate(alpha_input)
        c = alpha * c_s + (1 - alpha) * c_t

        # Final hidden
        h = o * torch.tanh(c)
        return h, c_s, c_t
