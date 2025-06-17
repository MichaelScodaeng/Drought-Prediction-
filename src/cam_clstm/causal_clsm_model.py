# === Enhanced ConvLSTM Integration Example ===
import torch
import torch.nn as nn
from src.cam_clstm.spatial_attention import EnhancedConvLSTMInputBlock

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)
        self.hidden_dim = hidden_dim

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)  # [B, C+H, H, W]
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()
        h = torch.zeros(B, self.cell.hidden_dim, H, W, device=x.device)
        c = torch.zeros(B, self.cell.hidden_dim, H, W, device=x.device)
        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1), (h, c)

class CausalAttentionBlock(nn.Module):
    def __init__(self, num_features, tasks, hidden_dim=32):
        super().__init__()
        self.tasks = tasks
        self.attn_nets = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_features),
                nn.Sigmoid()
            )
            for task in tasks
        })

    def forward(self, x, task):
        """
        x: [B, T, C, H, W]
        Returns: attention-weighted x, same shape
        """
        B, T, C, H, W = x.shape
        x_flat = x.mean(dim=[1, 3, 4])  # [B, C] — global mean over time+space
        attn_weights = self.attn_nets[task](x_flat)  # [B, C]
        attn_weights = attn_weights.view(B, 1, C, 1, 1)  # broadcast to [B, T, C, H, W]
        return x * attn_weights

class MyConvLSTMModel(nn.Module):
    def __init__(self, input_channels, height, width, hidden_channels,
                 pos_channels=4, use_pos_enc=True, use_spatial_attn=True,
                 causal_masks=None, use_temporal_only=False):
        super().__init__()
        print("Initializing MyConvLSTMModel with input channels:", input_channels)

        self.use_temporal_only = use_temporal_only
        self.height = height
        self.width = width
        self.causal_masks = causal_masks
        self.criterion = nn.MSELoss()
        self.temporal_only_channels = 0
        self.use_soft_causal_attention = True  # or toggle via argument
        self.causal_attention = CausalAttentionBlock(
            num_features=input_channels,  # original feature channels before pos/temporal
            tasks=['pre', 'pet', 'spei']
        )
        total_input_channels = input_channels
        if use_pos_enc:
            total_input_channels += pos_channels

        self.input_block = EnhancedConvLSTMInputBlock(
            input_channels=total_input_channels,
            height=height,
            width=width,
            pos_channels=0,
            use_pos_enc=False,
            use_spatial_attn=use_spatial_attn
        )

        self.convlstm = ConvLSTM(
            input_dim=total_input_channels,
            hidden_dim=hidden_channels
        )

        self.head_pre = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.head_pet = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.head_spei = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        

    def forward(self, x, y_true=None, loss_weights=None, temporal_only_input=None):
        """
        x: [B, T, C, H, W]
        temporal_only_input: [B, T, C_temp] (optional)
        y_true: dict of target tensors [B, 1, H, W] per task
        loss_weights: dict of task weights
        """
        B, T, _, H, W = x.shape
        preds = {}
        total_loss = 0.0
        losses = {}
        print("temporal_only_input shape:", temporal_only_input.shape if temporal_only_input is not None else "None")

        # Add temporal-only inputs
        if self.use_temporal_only and temporal_only_input is not None:
            if temporal_only_input.ndim == 2:
                temporal_only_input = temporal_only_input.unsqueeze(-1)

            B, T, C_temp = temporal_only_input.shape
            self.temporal_only_channels = C_temp
            temp_expanded = temporal_only_input.unsqueeze(-1).unsqueeze(-1)  # [B, T, C, 1, 1]
            temp_broadcasted = temp_expanded.expand(-1, -1, -1, H, W)        # [B, T, C, H, W]
            x = torch.cat([x, temp_broadcasted], dim=2)

        for key in ['pre', 'pet', 'spei']:
            # Apply per-task causal mask
            if self.use_soft_causal_attention:
                x_task = self.causal_attention(x, key)
            else:
                x_task = self._apply_causal_mask(x, key)

            # Positional encoding + spatial attention
            x_proc = self.input_block(x_task)           # [B, T, C', H, W]
            conv_out, _ = self.convlstm(x_proc)         # [B, T, hidden, H, W]
            last_out = conv_out[:, -1]                  # [B, hidden, H, W]

            head = getattr(self, f"head_{key}")
            preds[key] = head(last_out)                 # [B, 1, H, W]

            # Compute loss if ground truth is available
            if y_true and key in y_true:
                task_loss = self.criterion(preds[key], y_true[key])
                weight = loss_weights.get(key, 1.0) if loss_weights else 1.0
                total_loss += weight * task_loss
                losses[key] = task_loss.item()

        if y_true:
            return preds, total_loss, losses
        return preds




# === Usage Example ===
if __name__ == '__main__':
    B, T, C, H, W = 2, 12, 10, 16, 16
    x = torch.randn(B, T, C, H, W)
    y_true = {
        'pre': torch.randn(B, 1, H, W),
        'pet': torch.randn(B, 1, H, W),
        'spei': torch.randn(B, 1, H, W)
    }

    causal_masks = {
    'pre':  torch.tensor([1, 1, 0, 1, 0, 0, 1, 1, 0, 1]),
    'pet':  torch.tensor([1, 0, 1, 1, 1, 0, 0, 1, 1, 0]),
    'spei': torch.tensor([1, 0, 1, 1, 0, 1, 1, 1, 0, 1])
    }


    model = MyConvLSTMModel(
        input_channels=C,
        height=H,
        width=W,
        hidden_channels=32,
        pos_channels=4,
        use_pos_enc=True,
        use_spatial_attn=True,
        causal_masks=causal_masks
    )

    preds, total_loss, loss_dict = model(x, y_true)
    print("Output shapes:", {k: v.shape for k, v in preds.items()})
    print("Total Loss:", total_loss.item())
    print("Per-task Losses:", loss_dict)
