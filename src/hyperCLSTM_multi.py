import pandas as pd
import numpy as np
import yaml
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from tqdm import tqdm
import math


# --- Helper function for JSON serialization ---
def _to_python_type(obj):
    if isinstance(obj, dict): return {k: _to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [_to_python_type(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64)): return int(obj)
    elif isinstance(obj, (np.floating, np.float64)): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    else: return obj

# --- Deep Learning & Tuning Imports ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
    import optuna
    PYTORCH_AVAILABLE = True
    print("PyTorch, PyTorch Lightning, and Optuna successfully imported.")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch, PyTorch Lightning or Optuna not found. Please install them to run this pipeline.")

# --- Utility Function Imports ---
try:
    from src.data_utils import load_config, load_and_prepare_data
    from src.grid_utils import create_gridded_data
    print("ConvLSTM Pipeline: Successfully imported utility functions.")
except ImportError as e:
    print(f"ConvLSTM Pipeline Error: Could not import utility functions: {e}")

# --- IMPROVED ConvLSTM IMPLEMENTATION ---

def init_weights(module):
    """Initialize weights using Xavier/He initialization"""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

class ImprovedConvLSTMCell(nn.Module):
    """Enhanced ConvLSTM Cell with better initialization and optional batch norm"""
    
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, batch_norm=False):
        super(ImprovedConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.batch_norm = batch_norm
        
        # Handle both odd and even kernel sizes
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        
        # Single convolution for all gates (more efficient)
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
        
        # Optional batch normalization
        if batch_norm:
            self.bn = nn.BatchNorm2d(4 * hidden_dim)
        
        # Initialize weights properly
        self.apply(init_weights)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        # Optional batch normalization
        if self.batch_norm:
            combined_conv = self.bn(combined_conv)
        
        # Split into gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply activations
        i = torch.sigmoid(cc_i)    # Input gate
        f = torch.sigmoid(cc_f)    # Forget gate  
        o = torch.sigmoid(cc_o)    # Output gate
        g = torch.tanh(cc_g)       # Candidate values
        
        # Update cell and hidden states
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )

class ImprovedConvLSTM(nn.Module):
    """Enhanced ConvLSTM with flexible architecture and better memory management"""
    
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, 
                 batch_first=True, bias=True, return_all_layers=False, 
                 dropout=0.0, batch_norm=False):
        super(ImprovedConvLSTM, self).__init__()
        
        self._check_kernel_size_consistency(kernel_size)
        
        # Extend parameters for multilayer
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.dropout = dropout

        # Build layers
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(
                ImprovedConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=hidden_dim[i],
                    kernel_size=kernel_size[i],
                    bias=bias,
                    batch_norm=batch_norm
                )
            )
        
        self.cell_list = nn.ModuleList(cell_list)
        
        # Dropout layers
        if dropout > 0:
            self.dropout_layers = nn.ModuleList([
                nn.Dropout2d(dropout) for _ in range(num_layers - 1)
            ])

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        b, seq_len, _, h, w = input_tensor.size()
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w), device=input_tensor.device)
        
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c]
                )
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            
            # Apply dropout between layers (except last layer)
            if self.dropout > 0 and layer_idx < self.num_layers - 1:
                layer_output = self.dropout_layers[layer_idx](layer_output.view(-1, *layer_output.shape[2:])).view(layer_output.shape)
            
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size, device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or isinstance(kernel_size, int) or 
                (isinstance(kernel_size, list) and all([isinstance(elem, (tuple, int)) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple, int, or list of tuples/ints')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class MultitaskEncodingForecastingConvLSTM(nn.Module):
    """Advanced ConvLSTM with multitask learning for both single and multiple target variables"""
    
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, 
                 pre_seq_length, aft_seq_length, target_variables=['pre'], 
                 batch_first=True, dropout=0.1, batch_norm=False,
                 use_attention=False, multitask_fusion='concat'):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.target_variables = target_variables if isinstance(target_variables, list) else [target_variables]
        self.n_targets = len(self.target_variables)
        self.use_attention = use_attention
        self.multitask_fusion = multitask_fusion
        
        print(f"Initializing Multitask ConvLSTM for targets: {self.target_variables}")
        
        # Shared Encoder
        self.encoder = ImprovedConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=batch_first,
            return_all_layers=True,
            dropout=dropout,
            batch_norm=batch_norm
        )
        
        final_hidden_dim = hidden_dim[-1] if isinstance(hidden_dim, list) else hidden_dim
        
        # Multitask Architecture
        if self.n_targets == 1:
            # Single target (backward compatible)
            self.forecaster = ImprovedConvLSTM(
                input_dim=1,  # Single target
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                num_layers=num_layers,
                batch_first=batch_first,
                return_all_layers=False,
                dropout=dropout,
                batch_norm=batch_norm
            )
            
            # Single output head
            self.conv_out = nn.Sequential(
                nn.Conv2d(final_hidden_dim, final_hidden_dim // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(final_hidden_dim // 2, 1, kernel_size=1),
            )
            
            self.feature_proj = nn.Sequential(
                nn.Conv2d(final_hidden_dim, 1, kernel_size=1),
                nn.Tanh()
            )
            
        else:
            # Multiple targets (multitask learning)
            if multitask_fusion == 'separate':
                # Separate forecasters for each target
                self.forecasters = nn.ModuleDict()
                self.output_heads = nn.ModuleDict()
                self.feature_projections = nn.ModuleDict()
                
                for target in self.target_variables:
                    self.forecasters[target] = ImprovedConvLSTM(
                        input_dim=1,
                        hidden_dim=hidden_dim,
                        kernel_size=kernel_size,
                        num_layers=num_layers,
                        batch_first=batch_first,
                        return_all_layers=False,
                        dropout=dropout,
                        batch_norm=batch_norm
                    )
                    
                    self.output_heads[target] = nn.Sequential(
                        nn.Conv2d(final_hidden_dim, final_hidden_dim // 2, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(final_hidden_dim // 2, 1, kernel_size=1),
                    )
                    
                    self.feature_projections[target] = nn.Sequential(
                        nn.Conv2d(final_hidden_dim, 1, kernel_size=1),
                        nn.Tanh()
                    )
            
            elif multitask_fusion == 'shared':
                # Shared forecaster with multiple output heads
                self.forecaster = ImprovedConvLSTM(
                    input_dim=self.n_targets,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    batch_first=batch_first,
                    return_all_layers=False,
                    dropout=dropout,
                    batch_norm=batch_norm
                )
                
                # Separate output heads for each target
                self.output_heads = nn.ModuleDict()
                for target in self.target_variables:
                    self.output_heads[target] = nn.Sequential(
                        nn.Conv2d(final_hidden_dim, final_hidden_dim // 2, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(final_hidden_dim // 2, 1, kernel_size=1),
                    )
                
                # Shared feature projection
                self.feature_proj = nn.Sequential(
                    nn.Conv2d(final_hidden_dim, self.n_targets, kernel_size=1),
                    nn.Tanh()
                )
            
            else:  # 'concat' - default
                # Single forecaster with concatenated output
                self.forecaster = ImprovedConvLSTM(
                    input_dim=self.n_targets,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    batch_first=batch_first,
                    return_all_layers=False,
                    dropout=dropout,
                    batch_norm=batch_norm
                )
                
                # Single output head for all targets
                self.conv_out = nn.Sequential(
                    nn.Conv2d(final_hidden_dim, final_hidden_dim // 2, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(final_hidden_dim // 2, self.n_targets, kernel_size=1),
                )
                
                self.feature_proj = nn.Sequential(
                    nn.Conv2d(final_hidden_dim, self.n_targets, kernel_size=1),
                    nn.Tanh()
                )
        
        # Optional attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=final_hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # Initialize weights
        self.apply(init_weights)

    def forward(self, input_tensor, target_tensor=None, teacher_forcing_ratio=0.5):
        """
        Args:
            input_tensor: Input sequence (B, T_in, C, H, W)
            target_tensor: Target sequence for teacher forcing (B, T_out, n_targets, H, W) or (B, T_out, H, W) for single target
            teacher_forcing_ratio: Probability of using teacher forcing
        """
        batch_size = input_tensor.size(0)
        device = input_tensor.device
        
        # Encode input sequence
        encoder_outputs, encoder_states = self.encoder(input_tensor)
        encoder_final = encoder_outputs[-1]  # (B, T_in, H, H, W)
        
        if self.n_targets == 1:
            # Single target prediction (backward compatible)
            return self._forward_single_target(encoder_states, target_tensor, teacher_forcing_ratio)
        else:
            # Multitask prediction
            if self.multitask_fusion == 'separate':
                return self._forward_separate_targets(encoder_states, target_tensor, teacher_forcing_ratio)
            elif self.multitask_fusion == 'shared':
                return self._forward_shared_forecaster(encoder_states, target_tensor, teacher_forcing_ratio)
            else:  # 'concat'
                return self._forward_concat_targets(encoder_states, target_tensor, teacher_forcing_ratio)
    
    def _forward_single_target(self, encoder_states, target_tensor, teacher_forcing_ratio):
        """Single target forward pass (backward compatible)"""
        forecaster_states = encoder_states
        last_hidden = encoder_states[-1][0]
        forecaster_input = self.feature_proj(last_hidden).unsqueeze(1)
        
        predictions = []
        use_teacher_forcing = target_tensor is not None and torch.rand(1).item() < teacher_forcing_ratio
        
        for t in range(self.aft_seq_length):
            forecaster_output, forecaster_states = self.forecaster(forecaster_input, forecaster_states)
            hidden_state = forecaster_output[0][:, -1]
            pred = self.conv_out(hidden_state)
            predictions.append(pred)
            
            if use_teacher_forcing and target_tensor is not None and t < self.aft_seq_length - 1:
                if target_tensor.dim() == 4:  # (B, T, H, W)
                    next_input = target_tensor[:, t].unsqueeze(1).unsqueeze(1)
                else:  # (B, T, 1, H, W)
                    next_input = target_tensor[:, t].unsqueeze(1)
            else:
                next_input = pred.unsqueeze(1)
            
            forecaster_input = next_input
        
        return torch.stack(predictions, dim=1)  # (B, T_out, 1, H, W)
    
    def _forward_separate_targets(self, encoder_states, target_tensor, teacher_forcing_ratio):
        """Separate forecasters for each target"""
        predictions_dict = {}
        
        for i, target in enumerate(self.target_variables):
            forecaster_states = encoder_states
            last_hidden = encoder_states[-1][0]
            forecaster_input = self.feature_projections[target](last_hidden).unsqueeze(1)
            
            target_predictions = []
            use_teacher_forcing = target_tensor is not None and torch.rand(1).item() < teacher_forcing_ratio
            
            for t in range(self.aft_seq_length):
                forecaster_output, forecaster_states = self.forecasters[target](forecaster_input, forecaster_states)
                hidden_state = forecaster_output[0][:, -1]
                pred = self.output_heads[target](hidden_state)
                target_predictions.append(pred)
                
                if use_teacher_forcing and target_tensor is not None and t < self.aft_seq_length - 1:
                    if target_tensor.dim() == 5:  # (B, T, n_targets, H, W)
                        next_input = target_tensor[:, t, i:i+1].unsqueeze(1)
                    else:  # (B, T, H, W) - assume single target
                        next_input = target_tensor[:, t].unsqueeze(1).unsqueeze(1)
                else:
                    next_input = pred.unsqueeze(1)
                
                forecaster_input = next_input
            
            predictions_dict[target] = torch.stack(target_predictions, dim=1)
        
        # Concatenate all target predictions
        predictions_list = [predictions_dict[target] for target in self.target_variables]
        return torch.cat(predictions_list, dim=2)  # (B, T_out, n_targets, H, W)
    
    def _forward_shared_forecaster(self, encoder_states, target_tensor, teacher_forcing_ratio):
        """Shared forecaster with separate output heads"""
        forecaster_states = encoder_states
        last_hidden = encoder_states[-1][0]
        forecaster_input = self.feature_proj(last_hidden).unsqueeze(1)
        
        predictions_dict = {}
        use_teacher_forcing = target_tensor is not None and torch.rand(1).item() < teacher_forcing_ratio
        
        for t in range(self.aft_seq_length):
            forecaster_output, forecaster_states = self.forecaster(forecaster_input, forecaster_states)
            hidden_state = forecaster_output[0][:, -1]
            
            # Generate predictions for each target
            target_preds = []
            for target in self.target_variables:
                pred = self.output_heads[target](hidden_state)
                target_preds.append(pred)
            
            combined_pred = torch.cat(target_preds, dim=1)  # (B, n_targets, H, W)
            
            if t == 0:
                for i, target in enumerate(self.target_variables):
                    predictions_dict[target] = []
            
            for i, target in enumerate(self.target_variables):
                predictions_dict[target].append(target_preds[i])
            
            if use_teacher_forcing and target_tensor is not None and t < self.aft_seq_length - 1:
                if target_tensor.dim() == 5:  # (B, T, n_targets, H, W)
                    next_input = target_tensor[:, t].unsqueeze(1)
                else:  # (B, T, H, W) - single target
                    next_input = target_tensor[:, t].unsqueeze(1).unsqueeze(1)
                    next_input = next_input.expand(-1, -1, self.n_targets, -1, -1)
            else:
                next_input = combined_pred.unsqueeze(1)
            
            forecaster_input = next_input
        
        # Stack predictions for each target
        predictions_list = []
        for target in self.target_variables:
            predictions_list.append(torch.stack(predictions_dict[target], dim=1))
        
        return torch.cat(predictions_list, dim=2)  # (B, T_out, n_targets, H, W)
    
    def _forward_concat_targets(self, encoder_states, target_tensor, teacher_forcing_ratio):
        """Single forecaster with concatenated output"""
        forecaster_states = encoder_states
        last_hidden = encoder_states[-1][0]
        forecaster_input = self.feature_proj(last_hidden).unsqueeze(1)
        
        predictions = []
        use_teacher_forcing = target_tensor is not None and torch.rand(1).item() < teacher_forcing_ratio
        
        for t in range(self.aft_seq_length):
            forecaster_output, forecaster_states = self.forecaster(forecaster_input, forecaster_states)
            hidden_state = forecaster_output[0][:, -1]
            pred = self.conv_out(hidden_state)  # (B, n_targets, H, W)
            predictions.append(pred)
            
            if use_teacher_forcing and target_tensor is not None and t < self.aft_seq_length - 1:
                if target_tensor.dim() == 5:  # (B, T, n_targets, H, W)
                    next_input = target_tensor[:, t].unsqueeze(1)
                elif target_tensor.dim() == 4:  # (B, T, H, W) - single target
                    next_input = target_tensor[:, t].unsqueeze(1).unsqueeze(1)
                    next_input = next_input.expand(-1, -1, self.n_targets, -1, -1)
            else:
                next_input = pred.unsqueeze(1)
            
            forecaster_input = next_input
        
        return torch.stack(predictions, dim=1)  # (B, T_out, n_targets, H, W)

# --- Enhanced Lightning Module for Multitask Learning ---
class MultitaskGridModelLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate, mask, target_variables=['pre'], 
                 trial=None, weight_decay=1e-5, lr_scheduler='cosine', 
                 gradient_clip_val=1.0, teacher_forcing_ratio=0.5,
                 task_weights=None):
        super().__init__()
        
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.gradient_clip_val = gradient_clip_val
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.trial = trial
        self.target_variables = target_variables if isinstance(target_variables, list) else [target_variables]
        self.n_targets = len(self.target_variables)
        
        # Task weights for multitask learning
        if task_weights is None:
            self.task_weights = {target: 1.0 for target in self.target_variables}
        else:
            self.task_weights = task_weights
        
        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='none')
        self.mae_loss = nn.L1Loss(reduction='none')
        
        # Register mask as buffer
        self.register_buffer('mask', mask)
        
        # Storage for validation outputs
        self.validation_step_outputs = []
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model', 'trial', 'mask'])

    def forward(self, x, target=None):
        return self.model(x, target, self.teacher_forcing_ratio)

    def _calculate_multitask_masked_loss(self, y_hat, y, loss_fn=None):
        """Calculate masked loss for multitask learning"""
        if loss_fn is None:
            loss_fn = self.mse_loss
        
        total_loss = 0.0
        task_losses = {}
        
        if self.n_targets == 1:
            # Single target (backward compatible)
            if y_hat.dim() == 5 and y_hat.size(2) == 1:
                y_hat = y_hat.squeeze(2)
            if y.dim() == 4 and y.dim() == 4:
                pass
            elif y.dim() == 3 and y_hat.dim() == 4:
                if y_hat.size(1) == 1:
                    y_hat = y_hat.squeeze(1)
                else:
                    y = y.unsqueeze(1).expand_as(y_hat)
            
            loss = loss_fn(y_hat, y)
            if self.mask.dim() == 2:
                mask_expanded = self.mask.expand_as(loss)
            else:
                mask_expanded = self.mask
            
            masked_loss = loss * mask_expanded
            valid_pixels = mask_expanded.sum()
            if valid_pixels > 0:
                total_loss = masked_loss.sum() / valid_pixels
            else:
                total_loss = masked_loss.sum()
            
            task_losses[self.target_variables[0]] = total_loss
        
        else:
            # Multiple targets
            for i, target in enumerate(self.target_variables):
                # Extract target-specific predictions and ground truth
                pred_target = y_hat[:, :, i:i+1, :, :].squeeze(2)  # (B, T, H, W)
                gt_target = y[:, :, i, :, :] if y.dim() == 5 else y[:, :, :, :]  # (B, T, H, W)
                
                # Calculate loss for this target
                loss = loss_fn(pred_target, gt_target)
                
                # Apply mask
                if self.mask.dim() == 2:
                    mask_expanded = self.mask.expand_as(loss)
                else:
                    mask_expanded = self.mask
                
                masked_loss = loss * mask_expanded
                valid_pixels = mask_expanded.sum()
                
                if valid_pixels > 0:
                    task_loss = masked_loss.sum() / valid_pixels
                else:
                    task_loss = masked_loss.sum()
                
                task_losses[target] = task_loss
                total_loss += self.task_weights[target] * task_loss
        
        return total_loss, task_losses

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, y)  # Pass target for teacher forcing
        
        # Calculate multitask losses
        mse_loss, mse_task_losses = self._calculate_multitask_masked_loss(y_hat, y, self.mse_loss)
        mae_loss, mae_task_losses = self._calculate_multitask_masked_loss(y_hat, y, self.mae_loss)
        
        # Log overall metrics
        self.log('train_mse_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae_loss', mae_loss, on_step=True, on_epoch=True)
        self.log('train_rmse', torch.sqrt(mse_loss), on_step=True, on_epoch=True)
        
        # Log per-task metrics
        for target in self.target_variables:
            self.log(f'train_mse_loss_{target}', mse_task_losses[target], on_step=True, on_epoch=True)
            self.log(f'train_mae_loss_{target}', mae_task_losses[target], on_step=True, on_epoch=True)
            self.log(f'train_rmse_{target}', torch.sqrt(mse_task_losses[target]), on_step=True, on_epoch=True)
        
        return mse_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)  # No teacher forcing during validation
        
        # Calculate multitask losses
        mse_loss, mse_task_losses = self._calculate_multitask_masked_loss(y_hat, y, self.mse_loss)
        mae_loss, mae_task_losses = self._calculate_multitask_masked_loss(y_hat, y, self.mae_loss)
        
        # Log overall metrics
        self.log('val_loss', mse_loss, on_epoch=True, prog_bar=True)
        self.log('val_mae_loss', mae_loss, on_epoch=True)
        self.log('val_rmse', torch.sqrt(mse_loss), on_epoch=True, prog_bar=True)
        
        # Log per-task metrics
        for target in self.target_variables:
            self.log(f'val_mse_loss_{target}', mse_task_losses[target], on_epoch=True)
            self.log(f'val_mae_loss_{target}', mae_task_losses[target], on_epoch=True)
            self.log(f'val_rmse_{target}', torch.sqrt(mse_task_losses[target]), on_epoch=True)
        
        self.validation_step_outputs.append(mse_loss)
        return mse_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)  # No teacher forcing during prediction

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.validation_step_outputs.clear()
        
        # Report to Optuna if available
        if self.trial:
            self.trial.report(avg_loss.item(), self.current_epoch)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        if self.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )
            return [optimizer], [scheduler]
        elif self.lr_scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss'
                }
            }
        else:
            return optimizer

    def on_before_optimizer_step(self, optimizer):
        # Gradient clipping
        if self.gradient_clip_val > 0:
            self.clip_gradients(
                optimizer, 
                gradient_clip_val=self.gradient_clip_val, 
                gradient_clip_algorithm="norm"
            )

# --- Enhanced Dataset for Multitask Learning ---
class MultitaskSequenceDatasetConvLSTM(Dataset):
    def __init__(self, gridded_data, target_feature_indices, n_steps_in, n_steps_out=1, 
                 augment=False, noise_level=0.01):
        self.data = torch.tensor(gridded_data, dtype=torch.float32).permute(0, 3, 1, 2)
        self.target_feature_indices = target_feature_indices if isinstance(target_feature_indices, list) else [target_feature_indices]
        self.n_targets = len(self.target_feature_indices)
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.augment = augment
        self.noise_level = noise_level
        
    def __len__(self):
        return max(0, self.data.shape[0] - self.n_steps_in - self.n_steps_out + 1)
    
    def __getitem__(self, idx):
        end_idx = idx + self.n_steps_in
        out_end_idx = end_idx + self.n_steps_out
        
        seq_x = self.data[idx:end_idx]
        
        if self.n_targets == 1:
            # Single target (backward compatible)
            seq_y = self.data[end_idx:out_end_idx, self.target_feature_indices[0], :, :]
        else:
            # Multiple targets
            seq_y_list = []
            for target_idx in self.target_feature_indices:
                seq_y_list.append(self.data[end_idx:out_end_idx, target_idx, :, :])
            seq_y = torch.stack(seq_y_list, dim=1)  # (T_out, n_targets, H, W)
        
        # Data augmentation during training
        if self.augment:
            # Add small amount of Gaussian noise
            if torch.rand(1) < 0.5:
                noise = torch.randn_like(seq_x) * self.noise_level
                seq_x = seq_x + noise
            
            # Random horizontal flip
            if torch.rand(1) < 0.3:
                seq_x = torch.flip(seq_x, dims=[-1])
                seq_y = torch.flip(seq_y, dims=[-1])
        
        return seq_x, seq_y

# --- Enhanced Pipeline Class for Multitask Learning ---
class HyperMultitaskConvLSTMPipeline:
    def __init__(self, config_path="config.yaml"):
        self.config_path_abs = os.path.abspath(config_path)
        self.cfg = load_config(self.config_path_abs)
        
        self.experiment_name = self.cfg.get('project_setup', {}).get('experiment_name', 'multitask_convlstm_experiment')
        self.project_root_for_paths = os.path.join(os.path.dirname(self.config_path_abs),"..","..")
        self.run_output_dir = os.path.join(self.project_root_for_paths, 'run_outputs', self.experiment_name)
        self.run_models_dir = os.path.join(self.project_root_for_paths, 'models_saved', self.experiment_name)
        
        os.makedirs(self.run_output_dir, exist_ok=True)
        os.makedirs(self.run_models_dir, exist_ok=True)
        
        # Handle both single and multiple target variables
        target_var_config = self.cfg.get('project_setup', {}).get('target_variable', 'pre')
        if isinstance(target_var_config, str):
            self.target_variables = [target_var_config]
        else:
            self.target_variables = target_var_config
        
        self.is_multitask = len(self.target_variables) > 1
        
        print(f"Pipeline Mode: {'Multitask' if self.is_multitask else 'Single Task'}")
        print(f"Target Variables: {self.target_variables}")
        
        self.model = None
        self.all_metrics = {}
        self.mask = None
        self.full_df_raw = None
        self.gridded_data = None
        self.best_hyperparams = None
        self.target_scalers = {}

    def _get_abs_path_from_config_value(self, rp):
        return os.path.abspath(os.path.join(self.project_root_for_paths, rp)) if rp and not os.path.isabs(rp) else rp
    
    def _get_target_scalers(self):
        """Get scalers for all target variables"""
        try:
            from src.grid_utils import get_target_scaler
            scalers = {}
            for target in self.target_variables:
                scaler = get_target_scaler(self.full_df_raw, target)
                scalers[target] = scaler
                if scaler is not None:
                    print(f"Target scaler obtained for {target}")
                else:
                    print(f"Warning: No scaler found for {target}")
            return scalers
        except ImportError:
            print("Warning: Could not import get_target_scaler. Assuming no scaling was applied.")
            return {target: None for target in self.target_variables}
    
    def _inverse_transform_multitask_predictions(self, scaled_preds, scalers=None):
        """Inverse transform predictions for multiple targets"""
        if scalers is None or all(s is None for s in scalers.values()):
            return scaled_preds
        
        original_shape = scaled_preds.shape
        
        if isinstance(scaled_preds, torch.Tensor):
            scaled_preds_np = scaled_preds.cpu().numpy()
        else:
            scaled_preds_np = scaled_preds
        
        # Handle different dimensions
        if scaled_preds_np.ndim == 5:  # (B, T, n_targets, H, W)
            batch_size, time_steps, n_targets, height, width = scaled_preds_np.shape
            transformed_preds = np.zeros_like(scaled_preds_np)
            
            for i, target in enumerate(self.target_variables):
                if i < n_targets and scalers.get(target) is not None:
                    target_data = scaled_preds_np[:, :, i, :, :].reshape(-1, 1)
                    try:
                        transformed_data = scalers[target].inverse_transform(target_data)
                        transformed_preds[:, :, i, :, :] = transformed_data.reshape(batch_size, time_steps, height, width)
                    except Exception as e:
                        print(f"Warning: Could not inverse transform {target}: {e}")
                        transformed_preds[:, :, i, :, :] = scaled_preds_np[:, :, i, :, :]
                else:
                    if i < n_targets:
                        transformed_preds[:, :, i, :, :] = scaled_preds_np[:, :, i, :, :]
            
            return transformed_preds
        
        elif scaled_preds_np.ndim == 4:  # Single target (B, T, H, W)
            target = self.target_variables[0]
            if scalers.get(target) is not None:
                flat_preds = scaled_preds_np.reshape(-1, 1)
                try:
                    transformed_preds = scalers[target].inverse_transform(flat_preds)
                    return transformed_preds.reshape(original_shape)
                except Exception as e:
                    print(f"Warning: Could not inverse transform {target}: {e}")
            return scaled_preds_np
        
        return scaled_preds_np
    
    def _inverse_transform_multitask_actuals(self, scaled_actuals, scalers=None):
        """Inverse transform actual values for multiple targets"""
        if scalers is None or all(s is None for s in scalers.values()):
            return scaled_actuals
        
        original_shape = scaled_actuals.shape
        
        if isinstance(scaled_actuals, torch.Tensor):
            scaled_actuals_np = scaled_actuals.cpu().numpy()
        else:
            scaled_actuals_np = scaled_actuals
        
        # Handle different dimensions
        if scaled_actuals_np.ndim == 5:  # (B, T, n_targets, H, W)
            batch_size, time_steps, n_targets, height, width = scaled_actuals_np.shape
            transformed_actuals = np.zeros_like(scaled_actuals_np)
            
            for i, target in enumerate(self.target_variables):
                if i < n_targets and scalers.get(target) is not None:
                    target_data = scaled_actuals_np[:, :, i, :, :].reshape(-1, 1)
                    try:
                        transformed_data = scalers[target].inverse_transform(target_data)
                        transformed_actuals[:, :, i, :, :] = transformed_data.reshape(batch_size, time_steps, height, width)
                    except Exception as e:
                        print(f"Warning: Could not inverse transform {target}: {e}")
                        transformed_actuals[:, :, i, :, :] = scaled_actuals_np[:, :, i, :, :]
                else:
                    if i < n_targets:
                        transformed_actuals[:, :, i, :, :] = scaled_actuals_np[:, :, i, :, :]
            
            return transformed_actuals
        
        elif scaled_actuals_np.ndim == 4:  # Single target (B, T, H, W)
            target = self.target_variables[0]
            if scalers.get(target) is not None:
                flat_actuals = scaled_actuals_np.reshape(-1, 1)
                try:
                    transformed_actuals = scalers[target].inverse_transform(flat_actuals)
                    return transformed_actuals.reshape(original_shape)
                except Exception as e:
                    print(f"Warning: Could not inverse transform {target}: {e}")
            return scaled_actuals_np
        
        return scaled_actuals_np

    def _calculate_multitask_masked_metrics(self, actuals, preds, mask, scalers=None):
        """Calculate comprehensive metrics for multitask learning with inverse transformation"""
        
        # Apply inverse transformation
        if scalers is not None:
            print("Applying inverse transformation to predictions and actuals...")
            preds = self._inverse_transform_multitask_predictions(preds, scalers)
            actuals = self._inverse_transform_multitask_actuals(actuals, scalers)
            
            # Convert back to tensors if needed
            if not isinstance(preds, torch.Tensor):
                preds = torch.tensor(preds, dtype=torch.float32)
            if not isinstance(actuals, torch.Tensor):
                actuals = torch.tensor(actuals, dtype=torch.float32)
        
        metrics = {}
        mask_bool = mask.bool().to(actuals.device)
        
        if len(self.target_variables) == 1:
            # Single target (backward compatible)
            if preds.dim() == 5 and preds.size(2) == 1:
                preds = preds.squeeze(2)
            if actuals.dim() == 4 and actuals.size(1) == 1:
                actuals = actuals.squeeze(1)
            
            # Expand mask to match data dimensions
            if mask_bool.dim() == 2:
                if actuals.dim() == 4:  # (B, T, H, W)
                    batch_mask = mask_bool.unsqueeze(0).unsqueeze(0).expand_as(actuals)
                else:  # (B, H, W)
                    batch_mask = mask_bool.unsqueeze(0).expand_as(actuals)
            else:
                batch_mask = mask_bool.expand_as(actuals)
            
            actuals_masked = actuals[batch_mask]
            preds_masked = preds[batch_mask]
            
            actuals_np = actuals_masked.flatten().cpu().numpy()
            preds_np = preds_masked.flatten().cpu().numpy()
            
            target = self.target_variables[0]
            metrics[target] = {
                'rmse': np.sqrt(mean_squared_error(actuals_np, preds_np)),
                'mae': mean_absolute_error(actuals_np, preds_np),
                'r2': r2_score(actuals_np, preds_np),
                'mse': mean_squared_error(actuals_np, preds_np),
                'mape': np.mean(np.abs((actuals_np - preds_np) / (actuals_np + 1e-8))) * 100
            }
        
        else:
            # Multiple targets
            for i, target in enumerate(self.target_variables):
                # Extract target-specific data
                if preds.dim() == 5:  # (B, T, n_targets, H, W)
                    pred_target = preds[:, :, i, :, :]  # (B, T, H, W)
                    actual_target = actuals[:, :, i, :, :] if actuals.dim() == 5 else actuals
                else:  # (B, T, H, W) - single target
                    pred_target = preds
                    actual_target = actuals
                
                # Expand mask
                if mask_bool.dim() == 2:
                    if actual_target.dim() == 4:  # (B, T, H, W)
                        batch_mask = mask_bool.unsqueeze(0).unsqueeze(0).expand_as(actual_target)
                    else:  # (B, H, W)
                        batch_mask = mask_bool.unsqueeze(0).expand_as(actual_target)
                else:
                    batch_mask = mask_bool.expand_as(actual_target)
                
                actuals_masked = actual_target[batch_mask]
                preds_masked = pred_target[batch_mask]
                
                actuals_np = actuals_masked.flatten().cpu().numpy()
                preds_np = preds_masked.flatten().cpu().numpy()
                
                metrics[target] = {
                    'rmse': np.sqrt(mean_squared_error(actuals_np, preds_np)),
                    'mae': mean_absolute_error(actuals_np, preds_np),
                    'r2': r2_score(actuals_np, preds_np),
                    'mse': mean_squared_error(actuals_np, preds_np),
                    'mape': np.mean(np.abs((actuals_np - preds_np) / (actuals_np + 1e-8))) * 100
                }
        
        return metrics

    def _objective(self, trial, train_loader, val_loader, in_channels, pre_seq_len, aft_seq_len):
        """Enhanced Optuna objective for multitask learning"""
        cfg = self.cfg.get('convlstm_params', {}).get('tuning', {})
        
        # Hyperparameters to tune
        lr = trial.suggest_float('learning_rate', **cfg.get('learning_rate', {'low': 1e-4, 'high': 1e-2, 'log': True}))
        n_layers = trial.suggest_int('n_layers', **cfg.get('n_layers', {'low': 1, 'high': 3}))
        hidden_size = trial.suggest_categorical('hidden_dim_size', cfg.get('hidden_dim_size', {}).get('choices', [32, 64, 128]))
        kernel_size = trial.suggest_categorical('kernel_size', cfg.get('kernel_size', {}).get('choices', [3, 5]))
        dropout = trial.suggest_float('dropout', 0.0, 0.3)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        batch_norm = trial.suggest_categorical('batch_norm', [True, False])
        teacher_forcing_ratio = trial.suggest_float('teacher_forcing_ratio', 0.3, 0.8)
        # 🔥 NEW: Hyperbolic-specific hyperparameters
        base_curvature = trial.suggest_float('base_curvature', 0.5, 3.0)
        curvature_scaling = trial.suggest_categorical('curvature_scaling', ['linear', 'exponential', 'constant'])
        
        # Generate curvatures for each layer
        if curvature_scaling == 'linear':
            curvatures = [base_curvature * (i + 1) for i in range(n_layers)]
        elif curvature_scaling == 'exponential':
            curvatures = [base_curvature * (2 ** i) for i in range(n_layers)]
        else:  # constant
            curvatures = [base_curvature] * n_layers
        # Multitask-specific hyperparameters
        if self.is_multitask:
            multitask_fusion =  trial.suggest_categorical("multitask_fusion",["shared"])
            # Task weights (optional - can be tuned)
            task_weights = {}
            for target in self.target_variables:
                task_weights[target] = trial.suggest_float(f'weight_{target}', 0.5, 2.0)
        # Build model
        hidden_dim = [hidden_size] * n_layers
        kernel_size_tuple = (kernel_size, kernel_size)
        
        model = HyperbolicMultitaskEncodingForecastingConvLSTM(
            input_dim=in_channels,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size_tuple,
            num_layers=n_layers,
            pre_seq_length=pre_seq_len,
            aft_seq_length=aft_seq_len,
            target_variables=self.target_variables,
            dropout=dropout,
            batch_norm=batch_norm,
            multitask_fusion=multitask_fusion,
            curvatures=curvatures 
        )
        
        lightning_model = MultitaskGridModelLightningModule(
            model=model,
            learning_rate=lr,
            mask=self.mask,
            target_variables=self.target_variables,
            trial=trial,
            weight_decay=weight_decay,
            teacher_forcing_ratio=teacher_forcing_ratio,
            task_weights=task_weights
        )
        
        # Training configuration
        trainer_params = self.cfg.get('convlstm_params', {}).get('trainer', {})
        early_stopping = EarlyStopping(
            monitor="val_loss", 
            patience=trainer_params.get('patience_for_early_stopping', 10),
            min_delta=1e-6
        )
        
        trainer = pl.Trainer(
            max_epochs=self.cfg['convlstm_params']['tuning'].get('max_epochs', 40),
            callbacks=[early_stopping],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            accelerator='auto',
            devices=1,
            gradient_clip_val=1.0,
        )
        
        try:
            trainer.fit(
                model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader
            )
        except optuna.exceptions.TrialPruned:
            return float('inf')
        
        return trainer.callback_metrics.get("val_loss", torch.tensor(float('inf'))).item()

    def run_pipeline(self):
        """Run the complete multitask pipeline"""
        if not PYTORCH_AVAILABLE:
            return "Failed: Dependencies not found."
        
        print(f"\n--- Starting {'Multitask' if self.is_multitask else 'Single Task'} ConvLSTM Pipeline ---")
        
        # Load and prepare data
        raw_path = self._get_abs_path_from_config_value(self.cfg.get('data', {}).get('raw_data_path'))
        self.full_df_raw = load_and_prepare_data({
            'data': {
                'raw_data_path': raw_path,
                'time_column': self.cfg['data']['time_column']
            }
        })
        
        if self.full_df_raw is None:
            return "Failed: Data Load"
        
        print(f"Data loaded: {len(self.full_df_raw)} records")
        
        # Get target scalers for all variables
        self.target_scalers = self._get_target_scalers()
        
        # Create gridded data
        self.gridded_data, mask = create_gridded_data(self.full_df_raw, self.cfg)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        
        print(f"Grid shape: {self.gridded_data.shape}, Valid pixels: {mask.sum()}")
        
        # Split data
        time_steps = self.full_df_raw[self.cfg['data']['time_column']].unique()
        train_end_idx = np.where(time_steps <= np.datetime64(self.cfg['data']['train_end_date']))[0][-1]
        val_end_idx = np.where(time_steps <= np.datetime64(self.cfg['data']['validation_end_date']))[0][-1]
        
        train_grid = self.gridded_data[:train_end_idx+1]
        val_grid = self.gridded_data[train_end_idx+1:val_end_idx+1]
        test_grid = self.gridded_data[val_end_idx+1:]
        
        print(f"Train: {len(train_grid)}, Val: {len(val_grid)}, Test: {len(test_grid)} time steps")
        
        # Setup sequence parameters
        features_to_grid = self.cfg['data']['features_to_grid']
        target_indices = [features_to_grid.index(target) for target in self.target_variables]
        
        seq_cfg = self.cfg.get('sequence_params', {})
        n_in = seq_cfg.get('n_steps_in', 12)
        n_out = seq_cfg.get('n_steps_out', 1)
        
        # Create multitask datasets
        train_ds = MultitaskSequenceDatasetConvLSTM(
            train_grid, target_indices, n_in, n_out, augment=True
        )
        val_ds = MultitaskSequenceDatasetConvLSTM(
            val_grid, target_indices, n_in, n_out, augment=False
        )
        test_ds = MultitaskSequenceDatasetConvLSTM(
            test_grid, target_indices, n_in, n_out, augment=False
        )
        
        if len(train_ds) == 0 or len(val_ds) == 0:
            return "Failed: Not enough data for sequences"
        
        print(f"Dataset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        
        # Create data loaders
        batch_size = self.cfg.get('convlstm_params', {}).get('batch_size', 8)
        num_workers = 2 if os.name != 'nt' else 0
        
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, 
            num_workers=num_workers, pin_memory=True
        )
        
        # Hyperparameter optimization with Optuna
        print("\n--- Starting Hyperparameter Optimization ---")
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        n_trials = self.cfg.get('convlstm_params', {}).get('tuning', {}).get('n_trials', 20)
        study.optimize(
            lambda trial: self._objective(
                trial, train_loader, val_loader, 
                len(features_to_grid), n_in, n_out
            ),
            n_trials=n_trials,
            timeout=3600  # 1 hour timeout
        )
        
        self.best_hyperparams = study.best_trial.params
        print(f"Best hyperparameters found: {self.best_hyperparams}")
        
        # Train final model with best hyperparameters
        print("\n--- Training Final Model ---")
        best = self.best_hyperparams
        
        # Extract multitask parameters
        multitask_fusion = best.get('multitask_fusion', 'concat') if self.is_multitask else 'concat'
        task_weights = {}
        # 🔥 Extract hyperbolic parameters from optimization
        base_curvature = best.get('base_curvature', 1.0)
        curvature_scaling = best.get('curvature_scaling', 'linear')
        n_layers = best['n_layers']

        # Generate optimized curvatures
        if curvature_scaling == 'linear':
            curvatures = [base_curvature * (i + 1) for i in range(n_layers)]
        elif curvature_scaling == 'exponential':
            curvatures = [base_curvature * (2 ** i) for i in range(n_layers)]
        else:  # constant
            curvatures = [base_curvature] * n_layers

        print(f"Using optimized curvatures: {curvatures}")
        for target in self.target_variables:
            task_weights[target] = best.get(f'weight_{target}', 1.0)
        
        final_model = HyperbolicMultitaskEncodingForecastingConvLSTM(
            input_dim=len(features_to_grid),
            hidden_dim=[best['hidden_dim_size']] * best['n_layers'],
            kernel_size=(best['kernel_size'], best['kernel_size']),
            num_layers=best['n_layers'],
            pre_seq_length=n_in,
            aft_seq_length=n_out,
            target_variables=self.target_variables,
            dropout=best.get('dropout', 0.1),
            batch_norm=best.get('batch_norm', False),
            multitask_fusion=multitask_fusion,
            curvatures=curvatures
        )
        
        final_lightning_model = MultitaskGridModelLightningModule(
            model=final_model,
            learning_rate=best['learning_rate'],
            mask=self.mask,
            target_variables=self.target_variables,
            weight_decay=best.get('weight_decay', 1e-5),
            teacher_forcing_ratio=best.get('teacher_forcing_ratio', 0.5),
            task_weights=task_weights
        )
        
        # Combine train and validation for final training
        full_train_ds = torch.utils.data.ConcatDataset([train_ds, val_ds])
        full_train_loader = DataLoader(
            full_train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        
        # Setup callbacks and trainer
        ckpt_callback = ModelCheckpoint(
            dirpath=self.run_models_dir,
            filename="multitask-convlstm-best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=15,
            min_delta=1e-6,
            verbose=True
        )
        
        trainer_cfg = self.cfg.get('convlstm_params', {}).get('trainer', {})
        trainer_cfg = self.cfg['convlstm_params']['trainer']
        final_trainer = pl.Trainer(
            max_epochs=trainer_cfg.get('max_epochs', 100),
            callbacks=[ckpt_callback, lr_monitor, early_stopping],
            logger=True,
            enable_progress_bar=True,
            accelerator='auto',
            devices=1,
            gradient_clip_val=1.0,
            precision=16 if torch.cuda.is_available() else 32
        )
        # THIS IS THE MISSING CRITICAL STEP!
        print("Training final model with best hyperparameters...")
        final_trainer.fit(
            model=final_lightning_model,
            train_dataloaders=full_train_loader,
            val_dataloaders=val_loader  # Use original val_loader for monitoring
        )
        # Load best model
        best_model_path = ckpt_callback.best_model_path
        print(f"Best model saved at: {best_model_path}")
        
        self.model = MultitaskGridModelLightningModule.load_from_checkpoint(
            best_model_path,
            model=final_lightning_model.model,
            learning_rate=final_lightning_model.learning_rate,
            mask=self.mask,
            target_variables=self.target_variables
        )
        # Check if model path is None (safety check)
        if best_model_path is None:
            print("WARNING: No model was saved during training!")
            # Manual save as backup
            manual_path = os.path.join(self.run_models_dir, 'manual_final_model.ckpt')
            final_trainer.save_checkpoint(manual_path)
            best_model_path = manual_path
            print(f"Manually saved model to: {manual_path}")
        
        # Load the trained model
        self.model = MultitaskGridModelLightningModule.load_from_checkpoint(
            best_model_path,
            model=final_lightning_model.model,
            learning_rate=final_lightning_model.learning_rate,
            mask=self.mask,
            target_variables=self.target_variables
        )
    
        # Evaluate model
        self.evaluate_and_save(final_trainer, train_ds, val_ds, test_ds)
        
        # Generate predictions
        self.predict_on_full_data()
        
        # Save hyperparameters
        hyperparams_path = os.path.join(self.run_output_dir, 'best_hyperparameters.json')
        with open(hyperparams_path, 'w') as f:
            json.dump(_to_python_type(self.best_hyperparams), f, indent=4)
        
        print(f"\n--- {'Multitask' if self.is_multitask else 'Single Task'} ConvLSTM Pipeline Completed ---")
        print(f"Results saved to: {self.run_output_dir}")
        #save whole class instance
        import pickle

        with open(os.path.join(self.run_output_dir, 'pipeline_instance.pkl'), 'wb') as f:
            pickle.dump(self, f)
        
        return self.all_metrics

    def evaluate_and_save(self, trainer, train_dataset, val_dataset, test_dataset):
        """Enhanced evaluation for multitask learning with inverse transformation"""
        print(f"\n--- Comprehensive {'Multitask' if self.is_multitask else 'Single Task'} Model Evaluation ---")
        self.all_metrics = {}
        
        self.model.eval()
        
        with torch.no_grad():
            for split_name, dataset in [('train', train_dataset), ('validation', val_dataset), ('test', test_dataset)]:
                if len(dataset) == 0:
                    continue
                
                print(f"Evaluating {split_name} set...")
                
                loader = DataLoader(
                    dataset, 
                    batch_size=self.cfg.get('convlstm_params', {}).get('batch_size', 8),
                    num_workers=0
                )
                
                # Collect all predictions and actuals
                y_actual_list = []
                y_pred_list = []
                
                for batch in tqdm(loader, desc=f"Evaluating {split_name}"):
                    x, y = batch
                    
                    # Move to device if using GPU
                    if torch.cuda.is_available():
                        x, y = x.cuda(), y.cuda()
                        self.model = self.model.cuda()
                    
                    # Get predictions
                    with torch.no_grad():
                        pred = self.model(x)
                    
                    y_actual_list.append(y.cpu())
                    y_pred_list.append(pred.cpu())
                
                # Concatenate all batches
                y_actual_all = torch.cat(y_actual_list, dim=0)
                y_pred_all = torch.cat(y_pred_list, dim=0)
                
                # Calculate metrics with inverse transformation
                metrics = self._calculate_multitask_masked_metrics(
                    y_actual_all, y_pred_all, self.mask, scalers=self.target_scalers
                )
                self.all_metrics[split_name] = metrics
                
                # Print metrics
                print(f"  {split_name.capitalize()} Metrics (Original Units):")
                for target in self.target_variables:
                    target_metrics = metrics[target]
                    print(f"    {target.upper()}:")
                    print(f"      RMSE: {target_metrics['rmse']:.4f}")
                    print(f"      MAE:  {target_metrics['mae']:.4f}")
                    print(f"      R²:   {target_metrics['r2']:.4f}")
                    print(f"      MAPE: {target_metrics['mape']:.2f}%")
        
        # Save metrics
        metrics_filename = self.cfg.get('results', {}).get('metrics_filename', 'multitask_convlstm_metrics.json')
        metrics_path = os.path.join(self.run_output_dir, metrics_filename)
        
        with open(metrics_path, 'w') as f:
            json.dump(_to_python_type(self.all_metrics), f, indent=4)
        
        print(f"Evaluation metrics saved to: {metrics_path}")

    def predict_on_full_data(self):
        """Generate predictions on full dataset for multitask learning"""
        print(f"\n--- Generating Full Dataset Predictions ({'Multitask' if self.is_multitask else 'Single Task'}) ---")
        
        if self.model is None or self.gridded_data is None:
            print("Model or data not available for prediction")
            return None
        
        # Setup parameters
        features_to_grid = self.cfg['data']['features_to_grid']
        target_indices = [features_to_grid.index(target) for target in self.target_variables]
        
        seq_params = self.cfg.get('sequence_params', {})
        n_in = seq_params.get('n_steps_in', 12)
        n_out = seq_params.get('n_steps_out', 1)
        
        # Create full dataset
        full_dataset = MultitaskSequenceDatasetConvLSTM(
            self.gridded_data, target_indices, n_in, n_out, augment=False
        )
        
        if len(full_dataset) == 0:
            print("Not enough data for full prediction")
            return None
        
        print(f"Generating predictions for {len(full_dataset)} sequences...")
        
        # Create data loader
        full_loader = DataLoader(
            full_dataset,
            batch_size=self.cfg.get('convlstm_params', {}).get('batch_size', 8),
            num_workers=0,
            pin_memory=False
        )
        
        # Generate predictions
        self.model.eval()
        trainer = pl.Trainer(accelerator='auto', devices=1, logger=False, enable_progress_bar=True)
        
        with torch.no_grad():
            predicted_grids_list = trainer.predict(self.model, dataloaders=full_loader)
            predicted_grids = torch.cat(predicted_grids_list, dim=0).cpu().numpy()
        
        # Apply inverse transformation
        if any(scaler is not None for scaler in self.target_scalers.values()):
            print("Applying inverse transformation to predictions...")
            predicted_grids = self._inverse_transform_multitask_predictions(predicted_grids, self.target_scalers)
            print("Predictions successfully transformed back to original scale")
        else:
            print("Warning: No scalers available - predictions remain in scaled space")
        
        print(f"Prediction shape: {predicted_grids.shape}")
        
        # Un-grid predictions to create output DataFrame
        print("Converting grid predictions to CSV format...")
        
        time_steps = self.full_df_raw[self.cfg['data']['time_column']].unique()
        pred_start_time_idx = n_in
        prediction_times = time_steps[pred_start_time_idx:pred_start_time_idx + len(predicted_grids)]
        
        # Create coordinate mapping if not exists
        if 'row_idx' not in self.full_df_raw.columns or 'col_idx' not in self.full_df_raw.columns:
            grid_cfg = self.cfg.get('gridding', {})
            fixed_step = grid_cfg.get('fixed_step', 0.5)
            
            lat_min = self.full_df_raw[self.cfg['data']['lat_column']].min()
            lon_min = self.full_df_raw[self.cfg['data']['lon_column']].min()
            
            self.full_df_raw['row_idx'] = ((self.full_df_raw[self.cfg['data']['lat_column']] - lat_min) / fixed_step).round().astype(int)
            self.full_df_raw['col_idx'] = ((self.full_df_raw[self.cfg['data']['lon_column']] - lon_min) / fixed_step).round().astype(int)
        
        # Create coordinate lookup
        coord_cols = ['row_idx', 'col_idx', self.cfg['data']['lat_column'], self.cfg['data']['lon_column']]
        cell_to_coord = self.full_df_raw[coord_cols].drop_duplicates().set_index(['row_idx', 'col_idx'])
        
        # Get valid pixel indices
        valid_pixel_indices = np.argwhere(self.mask.cpu().numpy() == 1)
        
        # Generate output records
        output_records = []
        
        for i, pred_time in enumerate(tqdm(prediction_times, desc="Converting predictions")):
            if i >= len(predicted_grids):
                break
            
            # Handle different prediction shapes
            if predicted_grids.ndim == 5:  # (B, T, n_targets, H, W)
                pred_grids = predicted_grids[i, 0]  # (n_targets, H, W)
            elif predicted_grids.ndim == 4:  # (B, T, H, W) - single target
                pred_grids = predicted_grids[i, 0:1]  # (1, H, W)
            else:
                pred_grids = predicted_grids[i]
            
            # Get actual values for comparison
            actual_time_idx = pred_start_time_idx + i
            actual_grids = {}
            
            for j, target in enumerate(self.target_variables):
                target_idx = target_indices[j]
                
                if actual_time_idx < len(self.gridded_data):
                    actual_value_grid = self.gridded_data[actual_time_idx, :, :, target_idx]
                    
                    # Apply inverse transformation to actual values
                    if self.target_scalers.get(target) is not None:
                        actual_value_flat = actual_value_grid.reshape(-1, 1)
                        try:
                            actual_value_transformed = self.target_scalers[target].inverse_transform(actual_value_flat)
                            actual_grids[target] = actual_value_transformed.reshape(actual_value_grid.shape)
                        except Exception as e:
                            print(f"Warning: Could not inverse transform actual {target}: {e}")
                            actual_grids[target] = actual_value_grid
                    else:
                        actual_grids[target] = actual_value_grid
                else:
                    actual_grids[target] = None
            
            for r, c in valid_pixel_indices:
                try:
                    coords = cell_to_coord.loc[(r, c)]
                    lat_val = coords[self.cfg['data']['lat_column']]
                    lon_val = coords[self.cfg['data']['lon_column']]
                except KeyError:
                    continue
                
                # Create base record
                record = {
                    'time': pred_time,
                    'lat': lat_val,
                    'lon': lon_val,
                }
                
                # Add predictions and actuals for each target
                for j, target in enumerate(self.target_variables):
                    # Get predicted value
                    if pred_grids.ndim == 3:  # (n_targets, H, W)
                        predicted_value = pred_grids[j, r, c]
                    else:  # (H, W) - single target
                        predicted_value = pred_grids[r, c]
                    
                    # Get actual value
                    actual_value = np.nan
                    if actual_grids[target] is not None:
                        actual_value = actual_grids[target][r, c]
                    else:
                        # Try to find from original data
                        actual_row = self.full_df_raw[
                            (self.full_df_raw[self.cfg['data']['time_column']] == pred_time) &
                            (np.abs(self.full_df_raw[self.cfg['data']['lat_column']] - lat_val) < 0.01) &
                            (np.abs(self.full_df_raw[self.cfg['data']['lon_column']] - lon_val) < 0.01)
                        ]
                        
                        if not actual_row.empty:
                            actual_value = actual_row[target].iloc[0]
                    
                    # Add to record
                    record[target] = actual_value
                    record[f'{target}_predicted'] = predicted_value
                    record[f'{target}_prediction_error'] = predicted_value - actual_value if not np.isnan(actual_value) else np.nan
                    record[f'{target}_absolute_error'] = abs(predicted_value - actual_value) if not np.isnan(actual_value) else np.nan
                
                output_records.append(record)
        
        # Create output DataFrame
        output_df = pd.DataFrame(output_records)
        
        # Save predictions
        pred_filename = self.cfg.get('results', {}).get('predictions_filename', 'multitask_convlstm_predictions.csv')
        pred_path = os.path.join(self.run_output_dir, pred_filename)
        
        output_df.to_csv(pred_path, index=False)
        print(f"Full predictions saved to: {pred_path}")
        
        # Generate prediction summary for each target
        if not output_df.empty:
            summary_stats = {
                'total_predictions': len(output_df),
                'targets': {},
                'multitask_info': {
                    'n_targets': len(self.target_variables),
                    'target_variables': self.target_variables,
                    'fusion_method': getattr(self.model.model, 'multitask_fusion', 'concat')
                }
            }
            
            for target in self.target_variables:
                target_cols = [target, f'{target}_predicted']
                valid_predictions = output_df.dropna(subset=target_cols)
                
                if len(valid_predictions) > 0:
                    summary_stats['targets'][target] = {
                        'valid_predictions': len(valid_predictions),
                        'mean_actual': float(valid_predictions[target].mean()),
                        'mean_predicted': float(valid_predictions[f'{target}_predicted'].mean()),
                        'mean_absolute_error': float(valid_predictions[f'{target}_absolute_error'].mean()),
                        'rmse': float(np.sqrt((valid_predictions[f'{target}_prediction_error'] ** 2).mean())),
                        'r2_score': float(r2_score(
                            valid_predictions[target],
                            valid_predictions[f'{target}_predicted']
                        )),
                        'units': f'Original {target} units (inverse transformed)'
                    }
                else:
                    summary_stats['targets'][target] = {
                        'error': 'No valid predictions found'
                    }
            
            summary_path = os.path.join(self.run_output_dir, 'prediction_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(_to_python_type(summary_stats), f, indent=4)
            
            print(f"Prediction summary saved to: {summary_path}")
            
            # Print summary for each target
            for target in self.target_variables:
                if 'error' not in summary_stats['targets'][target]:
                    stats = summary_stats['targets'][target]
                    print(f"Summary for {target.upper()} (Original Units): "
                          f"MAE={stats['mean_absolute_error']:.4f}, "
                          f"RMSE={stats['rmse']:.4f}, "
                          f"R²={stats['r2_score']:.4f}")
        
        return output_df

# --- Utility Functions ---

def run_multitask_convlstm_pipeline(config_path="config.yaml"):
    """Convenience function to run the multitask pipeline"""
    pipeline = MultitaskConvLSTMPipeline(config_path)
    return pipeline.run_pipeline()

def load_multitask_trained_model(model_path, config_path="config.yaml"):
    """Load a trained multitask model for inference"""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch not available")
    
    cfg = load_config(config_path)
    
    # Create dummy mask for loading
    dummy_mask = torch.ones(10, 10)
    
    # Determine target variables
    target_var_config = cfg.get('project_setup', {}).get('target_variable', 'pre')
    if isinstance(target_var_config, str):
        target_variables = [target_var_config]
    else:
        target_variables = target_var_config
    
    model = MultitaskGridModelLightningModule.load_from_checkpoint(
        model_path,
        mask=dummy_mask,
        target_variables=target_variables,
        strict=False
    )
    
    return model

# Backward compatibility - single target functions
def run_improved_convlstm_pipeline(config_path="config.yaml"):
    """Backward compatible function for single target prediction"""
    return run_multitask_convlstm_pipeline(config_path)

def load_trained_model(model_path, config_path="config.yaml"):
    """Backward compatible function for loading models"""
    return load_multitask_trained_model(model_path, config_path)
def fix_pipeline_directories(config_file):
    """Fix the directory path issue in the pipeline"""
    
    # Get the actual directory where your config file is located
    config_dir = os.path.dirname(os.path.abspath(config_file))
    print(f"Config directory: {config_dir}")
    
    # Create safe output directories
    safe_dirs = {
        'run_outputs': os.path.join(config_dir, 'run_outputs'),
        'models_saved': os.path.join(config_dir, 'models_saved'), 
        'logs': os.path.join(config_dir, 'logs')
    }
    
    # Create directories with write permissions
    for name, path in safe_dirs.items():
        try:
            os.makedirs(path, exist_ok=True)
            
            # Test write permissions
            test_file = os.path.join(path, 'test.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"✅ {name}: {path}")
            
        except Exception as e:
            print(f"❌ Cannot create {name} at {path}: {e}")
            # Fallback to temp directory
            safe_dirs[name] = os.path.join(tempfile.gettempdir(), f'multitask_{name}')
            os.makedirs(safe_dirs[name], exist_ok=True)
            print(f"⚠️  Using temp directory for {name}: {safe_dirs[name]}")
    
    return safe_dirs

def run_pipeline_with_fixed_paths(config_file):
    """Run pipeline with corrected directory paths"""
    
    # Fix directories first
    safe_dirs = fix_pipeline_directories(config_file)
    
    # Import and create pipeline
    from improved_convlstm_multitask_pipeline import MultitaskConvLSTMPipeline
    
    # Create pipeline
    pipeline = MultitaskConvLSTMPipeline(config_file)
    
    # CRITICAL FIX: Override the problematic directory paths
    pipeline.run_output_dir = safe_dirs['run_outputs']
    pipeline.run_models_dir = safe_dirs['models_saved']
    
    print(f"Fixed paths:")
    print(f"  Output dir: {pipeline.run_output_dir}")
    print(f"  Models dir: {pipeline.run_models_dir}")
    
    # Now run the pipeline
    return pipeline.run_pipeline()
class HyperbolicOps:
    """Core hyperbolic operations in Poincaré ball model"""
    
    @staticmethod
    def exp_map(v, c=1.0, eps=1e-8):
        """Exponential map: Euclidean → Poincaré ball"""
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=eps)
        sqrt_c = math.sqrt(c)
        
        # Avoid NaN for zero vectors
        result = torch.where(
            v_norm > eps,
            torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm),
            v  # For zero vectors, return as-is
        )
        
        # Ensure we stay within Poincaré ball
        result_norm = torch.norm(result, dim=-1, keepdim=True)
        max_norm = (1.0 / sqrt_c) - eps
        result = torch.where(
            result_norm > max_norm,
            result * max_norm / result_norm,
            result
        )
        
        return result
    
    @staticmethod
    def log_map(x, c=1.0, eps=1e-8):
        """Logarithmic map: Poincaré ball → Euclidean"""
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=eps)
        sqrt_c = math.sqrt(c)
        
        # Clamp to avoid numerical issues
        clamped_norm = (sqrt_c * x_norm).clamp(max=1.0 - eps)
        
        result = torch.where(
            x_norm > eps,
            torch.atanh(clamped_norm) * x / (sqrt_c * x_norm),
            x  # For zero vectors
        )
        
        return result
    
    @staticmethod
    def mobius_add(x, y, c=1.0, eps=1e-8):
        """Möbius addition in Poincaré ball"""
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)
        xy_dot = torch.sum(x * y, dim=-1, keepdim=True)
        
        numerator = (1 + 2*c*xy_dot + c*y_norm_sq) * x + (1 - c*x_norm_sq) * y
        denominator = (1 + 2*c*xy_dot + c*c*x_norm_sq*y_norm_sq).clamp(min=eps)
        
        result = numerator / denominator
        
        # Ensure result stays in Poincaré ball
        result_norm = torch.norm(result, dim=-1, keepdim=True)
        max_norm = (1.0 / math.sqrt(c)) - eps
        result = torch.where(
            result_norm > max_norm,
            result * max_norm / result_norm,
            result
        )
        
        return result
    
    @staticmethod
    def hyperbolic_mul(x, y, c=1.0):
        """Element-wise multiplication in hyperbolic space (approximation)"""
        # Convert to Euclidean, multiply, convert back
        x_eucl = HyperbolicOps.log_map(x, c)
        y_eucl = HyperbolicOps.log_map(y, c)
        result_eucl = x_eucl * y_eucl
        return HyperbolicOps.exp_map(result_eucl, c)


class HyperbolicConvLSTMCell(nn.Module):
    """
    Hyperbolic ConvLSTM Cell - Drop-in replacement for ImprovedConvLSTMCell
    
    This cell processes LSTM gates in hyperbolic space for better hierarchical modeling.
    Maintains the same interface as your existing ImprovedConvLSTMCell.
    """
    
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, batch_norm=False, curvature=1.0):
        super(HyperbolicConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.batch_norm = batch_norm
        self.curvature = curvature
        
        # Handle both odd and even kernel sizes (same as your original)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        
        # Standard convolution for spatial processing (efficiency)
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
        
        # Optional batch normalization (same as your original)
        if batch_norm:
            self.bn = nn.BatchNorm2d(4 * hidden_dim)
        
        # Initialize weights using your existing function
        self.apply(init_weights)
    
    def forward(self, input_tensor, cur_state):
        """
        Forward pass - same interface as ImprovedConvLSTMCell
        
        Args:
            input_tensor: [batch, input_dim, height, width]
            cur_state: tuple of (h_cur, c_cur)
        
        Returns:
            (h_next, c_next): next hidden and cell states
        """
        h_cur, c_cur = cur_state
        
        # Standard spatial convolution (efficiency)
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        # Optional batch normalization
        if self.batch_norm:
            combined_conv = self.bn(combined_conv)
        
        # Split into gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # HYPERBOLIC INNOVATION: Process gates in hyperbolic space
        # Reshape for hyperbolic operations: [B, C, H, W] → [B, H, W, C]
        cc_i = cc_i.permute(0, 2, 3, 1)
        cc_f = cc_f.permute(0, 2, 3, 1) 
        cc_o = cc_o.permute(0, 2, 3, 1)
        cc_g = cc_g.permute(0, 2, 3, 1)
        
        # Convert current states to hyperbolic space
        h_cur_reshaped = h_cur.permute(0, 2, 3, 1)
        c_cur_reshaped = c_cur.permute(0, 2, 3, 1)
        
        # Hyperbolic gate activations
        i = HyperbolicOps.exp_map(torch.sigmoid(cc_i), self.curvature)  # Input gate
        f = HyperbolicOps.exp_map(torch.sigmoid(cc_f), self.curvature)  # Forget gate
        o = HyperbolicOps.exp_map(torch.sigmoid(cc_o), self.curvature)  # Output gate
        g = HyperbolicOps.exp_map(torch.tanh(cc_g), self.curvature)     # Candidate values
        
        # Project previous cell state to hyperbolic space
        c_cur_hyp = HyperbolicOps.exp_map(c_cur_reshaped, self.curvature)
        
        # Hyperbolic cell state update: C_t = f_t ⊗ C_{t-1} ⊕ i_t ⊗ g_t
        forget_term = HyperbolicOps.hyperbolic_mul(f, c_cur_hyp, self.curvature)
        input_term = HyperbolicOps.hyperbolic_mul(i, g, self.curvature)
        c_next_hyp = HyperbolicOps.mobius_add(forget_term, input_term, self.curvature)
        
        # Hyperbolic hidden state: H_t = o_t ⊗ tanh(C_t)
        c_activated = HyperbolicOps.exp_map(
            torch.tanh(HyperbolicOps.log_map(c_next_hyp, self.curvature)), 
            self.curvature
        )
        h_next_hyp = HyperbolicOps.hyperbolic_mul(o, c_activated, self.curvature)
        
        # Project back to Euclidean space for output
        h_next_eucl = HyperbolicOps.log_map(h_next_hyp, self.curvature)
        c_next_eucl = HyperbolicOps.log_map(c_next_hyp, self.curvature)
        
        # Reshape back: [B, H, W, C] → [B, C, H, W]
        h_next = h_next_eucl.permute(0, 3, 1, 2)
        c_next = c_next_eucl.permute(0, 3, 1, 2)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size, device):
        """Initialize hidden states - same interface as ImprovedConvLSTMCell"""
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )


class HyperbolicConvLSTM(nn.Module):
    """
    Hyperbolic ConvLSTM - Drop-in replacement for ImprovedConvLSTM
    
    Uses hyperbolic cells instead of standard cells for better hierarchical modeling.
    Maintains the same interface as your existing ImprovedConvLSTM.
    """
    
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, 
                 batch_first=True, bias=True, return_all_layers=False, 
                 dropout=0.0, batch_norm=False, curvatures=None):
        super(HyperbolicConvLSTM, self).__init__()
        
        self._check_kernel_size_consistency(kernel_size)
        
        # Extend parameters for multilayer (same as your original)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        
        # Default curvatures: increase with depth for hierarchy
        if curvatures is None:
            curvatures = [1.0 * (i + 1) for i in range(num_layers)]
        curvatures = self._extend_for_multilayer(curvatures, num_layers)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.dropout = dropout
        self.curvatures = curvatures
        
        # Build hyperbolic layers
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(
                HyperbolicConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=hidden_dim[i],
                    kernel_size=kernel_size[i],
                    bias=bias,
                    batch_norm=batch_norm,
                    curvature=curvatures[i]
                )
            )
        
        self.cell_list = nn.ModuleList(cell_list)
        
        # Dropout layers (same as your original)
        if dropout > 0:
            self.dropout_layers = nn.ModuleList([
                nn.Dropout2d(dropout) for _ in range(num_layers - 1)
            ])
    
    def forward(self, input_tensor, hidden_state=None):
        """Forward pass - same interface as ImprovedConvLSTM"""
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        b, seq_len, _, h, w = input_tensor.size()
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w), device=input_tensor.device)
        
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c]
                )
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            
            # Apply dropout between layers (same as your original)
            if self.dropout > 0 and layer_idx < self.num_layers - 1:
                layer_output = self.dropout_layers[layer_idx](layer_output.view(-1, *layer_output.shape[2:])).view(layer_output.shape)
            
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        
        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size, image_size, device):
        """Initialize hidden states"""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size, device))
        return init_states
    
    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """Same as your original"""
        if not (isinstance(kernel_size, tuple) or isinstance(kernel_size, int) or 
                (isinstance(kernel_size, list) and all([isinstance(elem, (tuple, int)) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple, int, or list of tuples/ints')
    
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """Same as your original"""
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class HyperbolicMultitaskEncodingForecastingConvLSTM(nn.Module):
    """
    Hyperbolic version of your MultitaskEncodingForecastingConvLSTM
    
    MINIMAL CHANGE: Just replace ImprovedConvLSTM with HyperbolicConvLSTM
    """
    
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, 
                 pre_seq_length, aft_seq_length, target_variables=['pre'], 
                 batch_first=True, dropout=0.1, batch_norm=False,
                 use_attention=False, multitask_fusion='concat', curvatures=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.target_variables = target_variables if isinstance(target_variables, list) else [target_variables]
        self.n_targets = len(self.target_variables)
        self.use_attention = use_attention
        self.multitask_fusion = multitask_fusion
        
        print(f"Initializing HYPERBOLIC Multitask ConvLSTM for targets: {self.target_variables}")
        print(f"Curvatures: {curvatures}")
        
        # MAIN CHANGE: Replace ImprovedConvLSTM with HyperbolicConvLSTM
        self.encoder = HyperbolicConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=batch_first,
            return_all_layers=True,
            dropout=dropout,
            batch_norm=batch_norm,
            curvatures=curvatures
        )
        
        final_hidden_dim = hidden_dim[-1] if isinstance(hidden_dim, list) else hidden_dim
        
        # Rest of the architecture remains exactly the same as your original
        if self.n_targets == 1:
            # Single target (backward compatible)
            self.forecaster = HyperbolicConvLSTM(
                input_dim=1,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                num_layers=num_layers,
                batch_first=batch_first,
                return_all_layers=False,
                dropout=dropout,
                batch_norm=batch_norm,
                curvatures=curvatures
            )
            
            self.conv_out = nn.Sequential(
                nn.Conv2d(final_hidden_dim, final_hidden_dim // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(final_hidden_dim // 2, 1, kernel_size=1),
            )
            
            self.feature_proj = nn.Sequential(
                nn.Conv2d(final_hidden_dim, 1, kernel_size=1),
                nn.Tanh()
            )
            
        else:
            # Multiple targets (multitask learning) - same as your original
            if multitask_fusion == 'separate':
                self.forecasters = nn.ModuleDict()
                self.output_heads = nn.ModuleDict()
                self.feature_projections = nn.ModuleDict()
                
                for target in self.target_variables:
                    self.forecasters[target] = HyperbolicConvLSTM(
                        input_dim=1,
                        hidden_dim=hidden_dim,
                        kernel_size=kernel_size,
                        num_layers=num_layers,
                        batch_first=batch_first,
                        return_all_layers=False,
                        dropout=dropout,
                        batch_norm=batch_norm,
                        curvatures=curvatures
                    )
                    
                    self.output_heads[target] = nn.Sequential(
                        nn.Conv2d(final_hidden_dim, final_hidden_dim // 2, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(final_hidden_dim // 2, 1, kernel_size=1),
                    )
                    
                    self.feature_projections[target] = nn.Sequential(
                        nn.Conv2d(final_hidden_dim, 1, kernel_size=1),
                        nn.Tanh()
                    )
            
            elif multitask_fusion == 'shared':
                self.forecaster = HyperbolicConvLSTM(
                    input_dim=self.n_targets,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    batch_first=batch_first,
                    return_all_layers=False,
                    dropout=dropout,
                    batch_norm=batch_norm,
                    curvatures=curvatures
                )
                
                self.output_heads = nn.ModuleDict()
                for target in self.target_variables:
                    self.output_heads[target] = nn.Sequential(
                        nn.Conv2d(final_hidden_dim, final_hidden_dim // 2, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(final_hidden_dim // 2, 1, kernel_size=1),
                    )
                
                self.feature_proj = nn.Sequential(
                    nn.Conv2d(final_hidden_dim, self.n_targets, kernel_size=1),
                    nn.Tanh()
                )
            
            else:  # 'concat'
                self.forecaster = HyperbolicConvLSTM(
                    input_dim=self.n_targets,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    batch_first=batch_first,
                    return_all_layers=False,
                    dropout=dropout,
                    batch_norm=batch_norm,
                    curvatures=curvatures
                )
                
                self.conv_out = nn.Sequential(
                    nn.Conv2d(final_hidden_dim, final_hidden_dim // 2, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(final_hidden_dim // 2, self.n_targets, kernel_size=1),
                )
                
                self.feature_proj = nn.Sequential(
                    nn.Conv2d(final_hidden_dim, self.n_targets, kernel_size=1),
                    nn.Tanh()
                )
        
        # Optional attention (same as original)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=final_hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # Initialize weights using your existing function
        self.apply(init_weights)
    
    # IMPORTANT: All forward methods remain exactly the same as your original
    # No changes needed to forward, _forward_single_target, etc.
    def forward(self, input_tensor, target_tensor=None, teacher_forcing_ratio=0.5):
        """Same as your original - no changes needed"""
        batch_size = input_tensor.size(0)
        device = input_tensor.device
        
        # Encode input sequence
        encoder_outputs, encoder_states = self.encoder(input_tensor)
        encoder_final = encoder_outputs[-1]
        
        if self.n_targets == 1:
            return self._forward_single_target(encoder_states, target_tensor, teacher_forcing_ratio)
        else:
            if self.multitask_fusion == 'separate':
                return self._forward_separate_targets(encoder_states, target_tensor, teacher_forcing_ratio)
            elif self.multitask_fusion == 'shared':
                return self._forward_shared_forecaster(encoder_states, target_tensor, teacher_forcing_ratio)
            else:
                return self._forward_concat_targets(encoder_states, target_tensor, teacher_forcing_ratio)
    
    # Copy all your existing _forward_* methods unchanged
    def _forward_single_target(self, encoder_states, target_tensor, teacher_forcing_ratio):
        """Copy from your original - no changes"""
        forecaster_states = encoder_states
        last_hidden = encoder_states[-1][0]
        forecaster_input = self.feature_proj(last_hidden).unsqueeze(1)
        
        predictions = []
        use_teacher_forcing = target_tensor is not None and torch.rand(1).item() < teacher_forcing_ratio
        
        for t in range(self.aft_seq_length):
            forecaster_output, forecaster_states = self.forecaster(forecaster_input, forecaster_states)
            hidden_state = forecaster_output[0][:, -1]
            pred = self.conv_out(hidden_state)
            predictions.append(pred)
            
            if use_teacher_forcing and target_tensor is not None and t < self.aft_seq_length - 1:
                if target_tensor.dim() == 4:
                    next_input = target_tensor[:, t].unsqueeze(1).unsqueeze(1)
                else:
                    next_input = target_tensor[:, t].unsqueeze(1)
            else:
                next_input = pred.unsqueeze(1)
            
            forecaster_input = next_input
        
        return torch.stack(predictions, dim=1)
    
    def _forward_separate_targets(self, encoder_states, target_tensor, teacher_forcing_ratio):
        """Copy from your original - no changes"""
        predictions_dict = {}
        
        for i, target in enumerate(self.target_variables):
            forecaster_states = encoder_states
            last_hidden = encoder_states[-1][0]
            forecaster_input = self.feature_projections[target](last_hidden).unsqueeze(1)
            
            target_predictions = []
            use_teacher_forcing = target_tensor is not None and torch.rand(1).item() < teacher_forcing_ratio
            
            for t in range(self.aft_seq_length):
                forecaster_output, forecaster_states = self.forecasters[target](forecaster_input, forecaster_states)
                hidden_state = forecaster_output[0][:, -1]
                pred = self.output_heads[target](hidden_state)
                target_predictions.append(pred)
                
                if use_teacher_forcing and target_tensor is not None and t < self.aft_seq_length - 1:
                    if target_tensor.dim() == 5:
                        next_input = target_tensor[:, t, i:i+1].unsqueeze(1)
                    else:
                        next_input = target_tensor[:, t].unsqueeze(1).unsqueeze(1)
                else:
                    next_input = pred.unsqueeze(1)
                
                forecaster_input = next_input
            
            predictions_dict[target] = torch.stack(target_predictions, dim=1)
        
        predictions_list = [predictions_dict[target] for target in self.target_variables]
        return torch.cat(predictions_list, dim=2)
    
    def _forward_shared_forecaster(self, encoder_states, target_tensor, teacher_forcing_ratio):
        """Copy from your original - no changes"""
        forecaster_states = encoder_states
        last_hidden = encoder_states[-1][0]
        forecaster_input = self.feature_proj(last_hidden).unsqueeze(1)
        
        predictions_dict = {}
        use_teacher_forcing = target_tensor is not None and torch.rand(1).item() < teacher_forcing_ratio
        
        for t in range(self.aft_seq_length):
            forecaster_output, forecaster_states = self.forecaster(forecaster_input, forecaster_states)
            hidden_state = forecaster_output[0][:, -1]
            
            target_preds = []
            for target in self.target_variables:
                pred = self.output_heads[target](hidden_state)
                target_preds.append(pred)
            
            combined_pred = torch.cat(target_preds, dim=1)
            
            if t == 0:
                for i, target in enumerate(self.target_variables):
                    predictions_dict[target] = []
            
            for i, target in enumerate(self.target_variables):
                predictions_dict[target].append(target_preds[i])
            
            if use_teacher_forcing and target_tensor is not None and t < self.aft_seq_length - 1:
                if target_tensor.dim() == 5:
                    next_input = target_tensor[:, t].unsqueeze(1)
                else:
                    next_input = target_tensor[:, t].unsqueeze(1).unsqueeze(1)
                    next_input = next_input.expand(-1, -1, self.n_targets, -1, -1)
            else:
                next_input = combined_pred.unsqueeze(1)
            
            forecaster_input = next_input
        
        predictions_list = []
        for target in self.target_variables:
            predictions_list.append(torch.stack(predictions_dict[target], dim=1))
        
        return torch.cat(predictions_list, dim=2)
    
    def _forward_concat_targets(self, encoder_states, target_tensor, teacher_forcing_ratio):
        """Copy from your original - no changes"""
        forecaster_states = encoder_states
        last_hidden = encoder_states[-1][0]
        forecaster_input = self.feature_proj(last_hidden).unsqueeze(1)
        
        predictions = []
        use_teacher_forcing = target_tensor is not None and torch.rand(1).item() < teacher_forcing_ratio
        
        for t in range(self.aft_seq_length):
            forecaster_output, forecaster_states = self.forecaster(forecaster_input, forecaster_states)
            hidden_state = forecaster_output[0][:, -1]
            pred = self.conv_out(hidden_state)
            predictions.append(pred)
            
            if use_teacher_forcing and target_tensor is not None and t < self.aft_seq_length - 1:
                if target_tensor.dim() == 5:
                    next_input = target_tensor[:, t].unsqueeze(1)
                elif target_tensor.dim() == 4:
                    next_input = target_tensor[:, t].unsqueeze(1).unsqueeze(1)
                    next_input = next_input.expand(-1, -1, self.n_targets, -1, -1)
            else:
                next_input = pred.unsqueeze(1)
            
            forecaster_input = next_input
        
        return torch.stack(predictions, dim=1)