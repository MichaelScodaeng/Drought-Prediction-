# stacked_model.py
import torch
import torch.nn as nn
from src.adm_prednet.cell import ADMConvLSTMCell

class ADMStackedModel(nn.Module):
    def __init__(self, input_channels, hidden_channels, n_layers, output_targets=["PET"], kernel_size=3, dropout_rate=0.3):
        super().__init__()
        assert isinstance(hidden_channels, list) and len(hidden_channels) == n_layers

        self.n_layers = n_layers
        self.output_targets = output_targets
        self.hidden_channels = hidden_channels
        self.cells = nn.ModuleList()

        for i in range(n_layers):
            in_ch = input_channels if i == 0 else hidden_channels[i-1]
            # Pass dropout_rate to each ADMConvLSTMCell
            self.cells.append(ADMConvLSTMCell(in_ch, hidden_channels[i], kernel_size, dropout_rate=dropout_rate))

        self.output_head = nn.Conv2d(
            hidden_channels[-1], len(output_targets), kernel_size=1
        )

    def forward(self, x_seq):
        """
        x_seq: [B, T, C, H, W] input sequence
        Returns: [B, len(output_targets), H, W]
        """
        B, T, C, H, W = x_seq.shape

        # Initialize hidden and memory states
        h = [torch.zeros(B, hc, H, W, device=x_seq.device) for hc in self.hidden_channels]
        c_s = [torch.zeros_like(h_i) for h_i in h]
        c_t = [torch.zeros_like(h_i) for h_i in h]

        for t in range(T):
            x_t = x_seq[:, t]  # [B, C, H, W]

            # Apply residual connections by adding previous layer's output to current input
            for l in range(self.n_layers):
                cell = self.cells[l]
                
                # Adding the residual connection from the previous layer
                if l > 0:
                    x_t = x_t + h[l-1]  # Adding the previous layer's output as a residual

                x_t, c_s[l], c_t[l] = cell(x_t, h[l], c_s[l], c_t[l])  # Get new hidden state
                h[l] = x_t  # Update hidden state for the current layer

        # Output head to produce final predictions
        y = self.output_head(h[-1])  # Final output after all layers
        return y  # [B, len(output_targets), H, W]
