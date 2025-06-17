import pandas as pd
import numpy as np
import yaml
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from tqdm import tqdm
import math
from src.cam_clstm.gridded_multitask_seq2seq_dataset import GriddedMultitaskSeq2SeqDataset
from src.cam_clstm.causal_clsm_model import MyConvLSTMModel
from src.cam_clstm.causal_multitask_lightning_module import CausalMultitaskLightningModule
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

class AdvancedEncodingForecastingConvLSTM(nn.Module):
    """Advanced ConvLSTM with teacher forcing, attention, and better forecasting"""
    
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, 
                 pre_seq_length, aft_seq_length, n_targets=1, 
                 batch_first=True, dropout=0.1, batch_norm=False,
                 use_attention=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.n_targets = n_targets
        self.use_attention = use_attention
        
        # Encoder
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
        
        # Forecaster - uses target features as input
        self.forecaster = ImprovedConvLSTM(
            input_dim=n_targets,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=batch_first,
            return_all_layers=False,
            dropout=dropout,
            batch_norm=batch_norm
        )
        
        final_hidden_dim = hidden_dim[-1] if isinstance(hidden_dim, list) else hidden_dim
        
        # Output projection layers
        self.conv_out = nn.Sequential(
            nn.Conv2d(final_hidden_dim, final_hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_hidden_dim // 2, n_targets, kernel_size=1),
        )
        
        # Feature projection for forecaster input
        self.feature_proj = nn.Sequential(
            nn.Conv2d(final_hidden_dim, n_targets, kernel_size=1),
            nn.Tanh()  # Bounded output for stability
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
            target_tensor: Target sequence for teacher forcing (B, T_out, H, W)
            teacher_forcing_ratio: Probability of using teacher forcing
        """
        batch_size = input_tensor.size(0)
        device = input_tensor.device
        
        # Encode input sequence
        encoder_outputs, encoder_states = self.encoder(input_tensor)
        
        # Get the final encoder output for attention (if used)
        encoder_final = encoder_outputs[-1]  # (B, T_in, H, H, W)
        
        # Initialize forecaster with encoder final states
        forecaster_states = encoder_states
        
        # Get initial input for forecaster
        last_hidden = encoder_states[-1][0]  # (B, H, H, W)
        forecaster_input = self.feature_proj(last_hidden).unsqueeze(1)  # (B, 1, n_targets, H, W)
        
        predictions = []
        use_teacher_forcing = target_tensor is not None and torch.rand(1).item() < teacher_forcing_ratio
        
        for t in range(self.aft_seq_length):
            # Apply attention if enabled
            if self.use_attention and hasattr(self, 'attention'):
                # Reshape for attention
                B, T, C, H, W = encoder_final.shape
                encoder_flat = encoder_final.view(B, T, C * H * W).transpose(1, 2)  # (B, C*H*W, T)
                forecaster_flat = forecaster_input.view(B, 1, -1)  # (B, 1, n_targets*H*W)
                
                attended, _ = self.attention(
                    forecaster_flat, encoder_flat, encoder_flat
                )
                # Note: This is a simplified attention - in practice you'd want more sophisticated attention
            
            # Forecaster step
            forecaster_output, forecaster_states = self.forecaster(forecaster_input, forecaster_states)
            
            # Generate prediction
            hidden_state = forecaster_output[0][:, -1]  # (B, H, H, W)
            pred = self.conv_out(hidden_state)  # (B, n_targets, H, W)
            
            predictions.append(pred)
            
            # Prepare next input
            if use_teacher_forcing and t < self.aft_seq_length - 1:
                # Use ground truth
                next_input = target_tensor[:, t].unsqueeze(1).unsqueeze(1)  # (B, 1, 1, H, W)
                if next_input.size(2) != self.n_targets:
                    next_input = next_input.expand(-1, -1, self.n_targets, -1, -1)
            else:
                # Use prediction
                next_input = pred.unsqueeze(1)  # (B, 1, n_targets, H, W)
            
            forecaster_input = next_input
        
        return torch.stack(predictions, dim=1)  # (B, T_out, n_targets, H, W)

# --- Enhanced Lightning Module with Advanced Features ---
class AdvancedGridModelLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate, mask, trial=None, 
                 weight_decay=1e-5, lr_scheduler='cosine', 
                 gradient_clip_val=1.0, teacher_forcing_ratio=0.5):
        super().__init__()
        
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.gradient_clip_val = gradient_clip_val
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.trial = trial
        
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

    def _calculate_masked_loss(self, y_hat, y, loss_fn=None):
        """Calculate masked loss with proper dimension handling"""
        if loss_fn is None:
            loss_fn = self.mse_loss
        
        # Handle dimension mismatches
        if y_hat.dim() == 5 and y_hat.size(2) == 1:
            y_hat = y_hat.squeeze(2)  # Remove channel dimension
        
        if y.dim() == 4 and y_hat.dim() == 4:
            # Both are (B, T, H, W)
            pass
        elif y.dim() == 3 and y_hat.dim() == 4:
            # y is (B, H, W), y_hat is (B, T, H, W)
            if y_hat.size(1) == 1:
                y_hat = y_hat.squeeze(1)
            else:
                y = y.unsqueeze(1).expand_as(y_hat)
        
        # Calculate loss
        loss = loss_fn(y_hat, y)
        
        # Apply mask
        if self.mask.dim() == 2:
            # Expand mask to match loss dimensions
            mask_expanded = self.mask.expand_as(loss)
        else:
            mask_expanded = self.mask
        
        masked_loss = loss * mask_expanded
        
        # Calculate mean over valid pixels
        valid_pixels = mask_expanded.sum()
        if valid_pixels > 0:
            return masked_loss.sum() / valid_pixels
        else:
            return masked_loss.sum()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, y)  # Pass target for teacher forcing
        
        # Calculate losses
        mse_loss = self._calculate_masked_loss(y_hat, y, self.mse_loss)
        mae_loss = self._calculate_masked_loss(y_hat, y, self.mae_loss)
        
        # Log metrics
        self.log('train_mse_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae_loss', mae_loss, on_step=True, on_epoch=True)
        self.log('train_rmse', torch.sqrt(mse_loss), on_step=True, on_epoch=True)
        
        return mse_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)  # No teacher forcing during validation
        
        # Calculate losses
        mse_loss = self._calculate_masked_loss(y_hat, y, self.mse_loss)
        mae_loss = self._calculate_masked_loss(y_hat, y, self.mae_loss)
        rmse = torch.sqrt(mse_loss)
        
        # Log metrics
        self.log('val_loss', mse_loss, on_epoch=True, prog_bar=True)
        self.log('val_mae_loss', mae_loss, on_epoch=True)
        self.log('val_rmse', rmse, on_epoch=True, prog_bar=True)
        
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

# --- Enhanced Dataset with Data Augmentation ---
class EnhancedSequenceDatasetConvLSTM(Dataset):
    def __init__(self, gridded_data, target_feature_idx, n_steps_in, n_steps_out=1, 
                 augment=False, noise_level=0.01):
        self.data = torch.tensor(gridded_data, dtype=torch.float32).permute(0, 3, 1, 2)
        self.target_feature_idx = target_feature_idx
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
        seq_y = self.data[end_idx:out_end_idx, self.target_feature_idx, :, :]
        
        # Data augmentation during training
        if self.augment:
            '''# Add small amount of Gaussian noise
            if torch.rand(1) < 0.5:
                noise = torch.randn_like(seq_x) * self.noise_level
                seq_x = seq_x + noise
            
            # Random horizontal flip
            if torch.rand(1) < 0.3:
                seq_x = torch.flip(seq_x, dims=[-1])
                seq_y = torch.flip(seq_y, dims=[-1])
            '''
            pass # Placeholder for future augmentations
        return seq_x, seq_y

# --- Enhanced Pipeline Class ---
class ImprovedConvLSTMPipeline:
    def __init__(self, config_path="config.yaml"):
        self.config_path_abs = os.path.abspath(config_path)
        self.cfg = load_config(self.config_path_abs)
        
        self.experiment_name = self.cfg.get('project_setup', {}).get('experiment_name', 'improved_convlstm_experiment')
        self.project_root_for_paths = os.path.join(os.path.dirname(self.config_path_abs),"..","..")
        self.run_output_dir = os.path.join(self.project_root_for_paths, 'run_outputs', self.experiment_name)
        self.run_models_dir = os.path.join(self.project_root_for_paths, 'models_saved', self.experiment_name)
        
        os.makedirs(self.run_output_dir, exist_ok=True)
        os.makedirs(self.run_models_dir, exist_ok=True)
        
        self.model = None
        self.all_metrics = {}
        self.mask = None
        self.full_df_raw = None
        self.gridded_data = None
        self.best_hyperparams = None

    def _get_abs_path_from_config_value(self, rp):
        return os.path.abspath(os.path.join(self.project_root_for_paths, rp)) if rp and not os.path.isabs(rp) else rp
    
    def _get_target_scaler(self):
        """Get the scaler for the target variable from grid_utils"""
        try:
            from src.grid_utils import get_target_scaler
            target_col = self.cfg['project_setup']['target_variable']
            return get_target_scaler(self.full_df_raw, target_col)
        except ImportError:
            print("Warning: Could not import get_target_scaler. Assuming no scaling was applied.")
            return None
    
    def _inverse_transform_predictions(self, scaled_preds, scaler=None):
        """Inverse transform predictions back to original scale"""
        if scaler is None:
            return scaled_preds
        
        # Handle different tensor dimensions
        original_shape = scaled_preds.shape
        
        if isinstance(scaled_preds, torch.Tensor):
            scaled_preds_np = scaled_preds.cpu().numpy()
        else:
            scaled_preds_np = scaled_preds
        
        # Flatten for inverse transform
        flat_preds = scaled_preds_np.reshape(-1, 1)
        
        # Inverse transform
        try:
            original_preds = scaler.inverse_transform(flat_preds)
            return original_preds.reshape(original_shape)
        except Exception as e:
            print(f"Warning: Could not inverse transform predictions: {e}")
            return scaled_preds_np
    
    def _inverse_transform_actuals(self, scaled_actuals, scaler=None):
        """Inverse transform actual values back to original scale"""
        if scaler is None:
            return scaled_actuals
        
        original_shape = scaled_actuals.shape
        
        if isinstance(scaled_actuals, torch.Tensor):
            scaled_actuals_np = scaled_actuals.cpu().numpy()
        else:
            scaled_actuals_np = scaled_actuals
        
        # Flatten for inverse transform
        flat_actuals = scaled_actuals_np.reshape(-1, 1)
        
        # Inverse transform
        try:
            original_actuals = scaler.inverse_transform(flat_actuals)
            return original_actuals.reshape(original_shape)
        except Exception as e:
            print(f"Warning: Could not inverse transform actuals: {e}")
            return scaled_actuals_np

    def _calculate_masked_metrics(self, actuals, preds, mask, scaler=None):
        """Calculate comprehensive metrics with proper masking and inverse transformation"""
        
        # CRITICAL: Inverse transform predictions and actuals back to original scale
        if scaler is not None:
            print("Applying inverse transformation to predictions and actuals...")
            preds = self._inverse_transform_predictions(preds, scaler)
            actuals = self._inverse_transform_actuals(actuals, scaler)
            
            # Convert back to tensors if needed
            if not isinstance(preds, torch.Tensor):
                preds = torch.tensor(preds, dtype=torch.float32)
            if not isinstance(actuals, torch.Tensor):
                actuals = torch.tensor(actuals, dtype=torch.float32)
        # Squeeze singleton channel dimension if present
        if preds.dim() == 4 and preds.shape[1] == 1:
            preds = preds.squeeze(1)
        if actuals.dim() == 4 and actuals.shape[1] == 1:
            actuals = actuals.squeeze(1)

        mask_bool = mask.bool().to(actuals.device)
        mask_bool = mask.bool().to(actuals.device)
        
        # Handle dimension alignment
        if preds.dim() == 5 and preds.size(2) == 1:
            preds = preds.squeeze(2)
        if preds.dim() == 4 and actuals.dim() == 3:
            preds = preds.squeeze(1)
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
        
        # Extract valid pixels
        actuals_masked = actuals[batch_mask]
        preds_masked = preds[batch_mask]
        
        # Convert to numpy
        actuals_np = actuals_masked.flatten().cpu().numpy()
        preds_np = preds_masked.flatten().cpu().numpy()
        
        # Calculate metrics
        mse = mean_squared_error(actuals_np, preds_np)
        mae = mean_absolute_error(actuals_np, preds_np)
        r2 = r2_score(actuals_np, preds_np)
        rmse = np.sqrt(mse)
        
        # Additional metrics
        mape = np.mean(np.abs((actuals_np - preds_np) / (actuals_np + 1e-8))) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mse': mse,
            'mape': mape
        }

    def _objective(self, trial, train_loader, val_loader, in_channels, pre_seq_len, aft_seq_len):
        """Enhanced Optuna objective with more hyperparameters"""
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
        
        # Build model
        hidden_dim = [hidden_size] * n_layers
        kernel_size_tuple = (kernel_size, kernel_size)
        
        model = AdvancedEncodingForecastingConvLSTM(
            input_dim=in_channels,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size_tuple,
            num_layers=n_layers,
            pre_seq_length=pre_seq_len,
            aft_seq_length=aft_seq_len,
            n_targets=1,
            dropout=dropout,
            batch_norm=batch_norm
        )
        
        lightning_model = AdvancedGridModelLightningModule(
            model=model,
            learning_rate=lr,
            mask=self.mask,
            trial=trial,
            weight_decay=weight_decay,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        
        # Training configuration
        trainer_params = self.cfg.get('convlstm_params', {}).get('trainer', {})
        early_stopping = EarlyStopping(
            monitor="val_loss", 
            patience=trainer_params.get('patience_for_early_stopping', 10),
            min_delta=1e-6
        )
        tuning_cfg = self.cfg.get('convlstm_params', {}).get('tuning', {})
        trainer = pl.Trainer(
            max_epochs=tuning_cfg.get('max_epochs', 50),
            callbacks=[early_stopping],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
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
        """Run the complete improved pipeline"""
        if not PYTORCH_AVAILABLE:
            return "Failed: Dependencies not found."
        
        print(f"\n--- Starting Improved ConvLSTM Pipeline ---")
        import joblib
        # Load and prepare data
        raw_path = self._get_abs_path_from_config_value(self.cfg.get('data', {}).get('raw_data_path'))
        scaler_path = self._get_abs_path_from_config_value(self.cfg.get('data', {}).get('scaler_path'))
        self.target_scaler = joblib.load(scaler_path)
        self.full_df_raw = load_and_prepare_data({
            'data': {
                'raw_data_path': raw_path,
                'time_column': self.cfg['data']['time_column']
            }
        })
        
        if self.full_df_raw is None:
            return "Failed: Data Load"
        
        print(f"Data loaded: {len(self.full_df_raw)} records")
        
        # CRITICAL: Get the target scaler for inverse transformation
        self.target_scaler = self._get_target_scaler()
        if self.target_scaler is not None:
            print("Target scaler obtained for inverse transformation")
        else:
            print("Warning: No target scaler found - predictions will remain in scaled space")
        
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
        target_col = self.cfg['project_setup']['target_variable']
        features_to_grid = self.cfg['data']['features_to_grid']
        target_idx = features_to_grid.index(target_col)
        
        seq_cfg = self.cfg.get('sequence_params', {})
        n_in = seq_cfg.get('n_steps_in', 12)
        n_out = seq_cfg.get('n_steps_out', 1)
        
        # Create enhanced datasets
        train_ds = EnhancedSequenceDatasetConvLSTM(
            train_grid, target_idx, n_in, n_out, augment=True
        )
        val_ds = EnhancedSequenceDatasetConvLSTM(
            val_grid, target_idx, n_in, n_out, augment=False
        )
        test_ds = EnhancedSequenceDatasetConvLSTM(
            test_grid, target_idx, n_in, n_out, augment=False
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
        
        final_model = AdvancedEncodingForecastingConvLSTM(
            input_dim=len(features_to_grid),
            hidden_dim=[best['hidden_dim_size']] * best['n_layers'],
            kernel_size=(best['kernel_size'], best['kernel_size']),
            num_layers=best['n_layers'],
            pre_seq_length=n_in,
            aft_seq_length=n_out,
            n_targets=1,
            dropout=best.get('dropout', 0.1),
            batch_norm=best.get('batch_norm', False)
        )
        
        final_lightning_model = AdvancedGridModelLightningModule(
            model=final_model,
            learning_rate=best['learning_rate'],
            mask=self.mask,
            weight_decay=best.get('weight_decay', 1e-5),
            teacher_forcing_ratio=best.get('teacher_forcing_ratio', 0.5)
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
            filename="improved-convlstm-best-{epoch:02d}-{val_loss:.4f}",
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
        tuning_cfg = self.cfg.get('convlstm_params', {}).get('tuning', {})
        final_trainer = pl.Trainer(
            max_epochs=trainer_cfg.get('max_epochs', 100),
            callbacks=[ckpt_callback, lr_monitor, early_stopping],
            logger=True,
            enable_progress_bar=True,
            accelerator='auto',
            devices=1,
            gradient_clip_val=1.0,
            precision=16 if torch.cuda.is_available() else 32  # Mixed precision if available
        )
        
        # Final training
        final_trainer.fit(
            model=final_lightning_model,
            train_dataloaders=full_train_loader,
            val_dataloaders=val_loader
        )
        
        # Load best model
        best_model_path = ckpt_callback.best_model_path
        print(f"Best model saved at: {best_model_path}")
        
        self.model = AdvancedGridModelLightningModule.load_from_checkpoint(
            best_model_path,
            model=final_lightning_model.model,
            learning_rate=final_lightning_model.learning_rate,
            mask=self.mask
        )
        
        # Evaluate model
        self.evaluate_and_save(final_trainer, train_ds, val_ds, test_ds)
        
        # Generate predictions
        self.predict_on_full_data()
        
        # Save hyperparameters
        hyperparams_path = os.path.join(self.run_output_dir, 'best_hyperparameters.json')
        with open(hyperparams_path, 'w') as f:
            json.dump(_to_python_type(self.best_hyperparams), f, indent=4)
        
        print(f"\n--- Improved ConvLSTM Pipeline Completed ---")
        print(f"Results saved to: {self.run_output_dir}")
        
        return self.all_metrics

    def evaluate_and_save(self, trainer, train_dataset, val_dataset, test_dataset):
        """Enhanced evaluation with comprehensive metrics and inverse transformation"""
        print("\n--- Comprehensive Model Evaluation ---")
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
                    num_workers=0  # Use 0 for evaluation to avoid multiprocessing issues
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
                
                # CRITICAL: Calculate metrics with inverse transformation
                metrics = self._calculate_masked_metrics(
                    y_actual_all, y_pred_all, self.mask, scaler=self.target_scaler
                )
                self.all_metrics[split_name] = metrics
                
                # Print metrics
                target_col = self.cfg['project_setup']['target_variable']
                print(f"  {split_name.capitalize()} Metrics (Original {target_col} Units):")
                print(f"    RMSE: {metrics['rmse']:.4f}")
                print(f"    MAE:  {metrics['mae']:.4f}")
                print(f"    R²:   {metrics['r2']:.4f}")
                print(f"    MAPE: {metrics['mape']:.2f}%")
        
        # Save metrics
        metrics_filename = self.cfg.get('results', {}).get('metrics_filename', 'improved_convlstm_metrics.json')
        metrics_path = os.path.join(self.run_output_dir, metrics_filename)
        
        with open(metrics_path, 'w') as f:
            json.dump(_to_python_type(self.all_metrics), f, indent=4)
        
        print(f"Evaluation metrics saved to: {metrics_path}")

    def predict_on_full_data(self):
        """Generate predictions on full dataset with improved handling"""
        print("\n--- Generating Full Dataset Predictions ---")
        
        if self.model is None or self.gridded_data is None:
            print("Model or data not available for prediction")
            return None
        
        # Setup parameters
        target_col = self.cfg['project_setup']['target_variable']
        features_to_grid = self.cfg['data']['features_to_grid']
        target_idx = features_to_grid.index(target_col)
        
        seq_params = self.cfg.get('sequence_params', {})
        n_in = seq_params.get('n_steps_in', 12)
        n_out = seq_params.get('n_steps_out', 1)
        
        # Create full dataset
        full_dataset = EnhancedSequenceDatasetConvLSTM(
            self.gridded_data, target_idx, n_in, n_out, augment=False
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
        trainer = pl.Trainer(accelerator='auto', devices=1, logger=False,
            enable_progress_bar=True,)
        
        with torch.no_grad():
            predicted_grids_list = trainer.predict(self.model, dataloaders=full_loader)
            predicted_grids = torch.cat(predicted_grids_list, dim=0).cpu().numpy()
        
        # CRITICAL: Inverse transform predictions back to original scale
        if self.target_scaler is not None:
            print("Applying inverse transformation to predictions...")
            predicted_grids = self._inverse_transform_predictions(predicted_grids, self.target_scaler)
            print("Predictions successfully transformed back to original scale")
        else:
            print("Warning: No scaler available - predictions remain in scaled space")
        
        # Handle dimensions
        if predicted_grids.ndim == 5:
            predicted_grids = np.squeeze(predicted_grids, axis=2)  # Remove channel dimension
        if predicted_grids.ndim == 4 and predicted_grids.shape[1] == 1:
            predicted_grids = np.squeeze(predicted_grids, axis=1)  # Remove time dimension if single step
        
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
                
            pred_grid = predicted_grids[i]
            
            # Get actual values for comparison
            actual_time_idx = pred_start_time_idx + i
            if actual_time_idx < len(self.gridded_data):
                actual_grid = self.gridded_data[actual_time_idx, :, :, target_idx]
            else:
                actual_grid = None
            
            for r, c in valid_pixel_indices:
                try:
                    coords = cell_to_coord.loc[(r, c)]
                    lat_val = coords[self.cfg['data']['lat_column']]
                    lon_val = coords[self.cfg['data']['lon_column']]
                except KeyError:
                    continue
                
                # Find actual value and apply inverse transformation if needed
                actual_value = np.nan
                if actual_time_idx < len(self.gridded_data):
                    # Get actual value from gridded data
                    actual_value = self.gridded_data[actual_time_idx, r, c, target_idx]
                    
                    # CRITICAL: Inverse transform actual value back to original scale
                    if self.target_scaler is not None:
                        actual_value_reshaped = np.array([[actual_value]])
                        try:
                            actual_value = self.target_scaler.inverse_transform(actual_value_reshaped)[0, 0]
                        except Exception as e:
                            print(f"Warning: Could not inverse transform actual value: {e}")
                else:
                    # Try to find from original data
                    actual_row = self.full_df_raw[
                        (self.full_df_raw[self.cfg['data']['time_column']] == pred_time) &
                        (np.abs(self.full_df_raw[self.cfg['data']['lat_column']] - lat_val) < 0.01) &
                        (np.abs(self.full_df_raw[self.cfg['data']['lon_column']] - lon_val) < 0.01)
                    ]
                    
                    if not actual_row.empty:
                        actual_value = actual_row[target_col].iloc[0]
                        # Note: Original data should already be in original scale
                
                # Get predicted value (already inverse transformed above)
                predicted_value = pred_grid[r, c]
                
                # Create record
                record = {
                    'time': pred_time,
                    'lat': lat_val,
                    'lon': lon_val,
                    target_col: actual_value,
                    f'{target_col}_predicted': predicted_value,
                    'prediction_error': predicted_value - actual_value if not np.isnan(actual_value) else np.nan,
                    'absolute_error': abs(predicted_value - actual_value) if not np.isnan(actual_value) else np.nan
                }
                
                output_records.append(record)
        
        # Create output DataFrame
        output_df = pd.DataFrame(output_records)
        
        # Save predictions
        pred_filename = self.cfg.get('results', {}).get('predictions_filename', 'improved_convlstm_predictions.csv')
        pred_path = os.path.join(self.run_output_dir, pred_filename)
        
        output_df.to_csv(pred_path, index=False)
        print(f"Full predictions saved to: {pred_path}")
        
        # Generate prediction summary
        if not output_df.empty:
            valid_predictions = output_df.dropna(subset=[target_col, f'{target_col}_predicted'])
            
            if len(valid_predictions) > 0:
                summary_stats = {
                    'total_predictions': len(output_df),
                    'valid_predictions': len(valid_predictions),
                    'mean_actual': float(valid_predictions[target_col].mean()),
                    'mean_predicted': float(valid_predictions[f'{target_col}_predicted'].mean()),
                    'mean_absolute_error': float(valid_predictions['absolute_error'].mean()),
                    'rmse': float(np.sqrt((valid_predictions['prediction_error'] ** 2).mean())),
                    'r2_score': float(r2_score(
                        valid_predictions[target_col],
                        valid_predictions[f'{target_col}_predicted']
                    )),
                    'units': f'Original {target_col} units (inverse transformed)'
                }
                
                summary_path = os.path.join(self.run_output_dir, 'prediction_summary.json')
                with open(summary_path, 'w') as f:
                    json.dump(_to_python_type(summary_stats), f, indent=4)
                
                print(f"Prediction summary saved to: {summary_path}")
                print(f"Summary (Original Units): MAE={summary_stats['mean_absolute_error']:.4f}, "
                      f"RMSE={summary_stats['rmse']:.4f}, R²={summary_stats['r2_score']:.4f}")
            else:
                print("Warning: No valid predictions found for summary statistics")
        else:
            print("Warning: No predictions generated")
        
        return output_df

# --- Utility Functions ---

def run_improved_convlstm_pipeline(config_path="config.yaml"):
    """Convenience function to run the improved pipeline"""
    pipeline = ImprovedConvLSTMPipeline(config_path)
    return pipeline.run_pipeline()

def load_trained_model(model_path, config_path="config.yaml"):
    """Load a trained model for inference"""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch not available")
    
    cfg = load_config(config_path)
    
    # Create dummy mask for loading (will be replaced)
    dummy_mask = torch.ones(10, 10)
    
    model = AdvancedGridModelLightningModule.load_from_checkpoint(
        model_path,
        mask=dummy_mask,
        strict=False  # Allow loading with different mask
    )
    
    return model

# Example usage
if __name__ == "__main__":
    # Run the improved pipeline
    pipeline = ImprovedConvLSTMPipeline("config.yaml")
    results = pipeline.run_pipeline()
    
    print("Pipeline completed successfully!")
    print("Final metrics:", results)