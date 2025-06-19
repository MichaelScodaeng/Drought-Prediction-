from typing import Dict, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass
from src.mesanet.state_machine import MemoryState, MemoryConfig, StateTransitionNetwork
class MESANetLoss(nn.Module):
    """Unified loss function for MESA-Net"""
    
    def __init__(self, 
                 alpha_prediction: float = 1.0,
                 alpha_state_entropy: float = 0.1,
                 alpha_transition_smooth: float = 0.01,
                 alpha_cross_memory: float = 0.05,
                 alpha_cross_layer: float = 0.05):
        super().__init__()
        
        self.alpha_prediction = alpha_prediction
        self.alpha_state_entropy = alpha_state_entropy
        self.alpha_transition_smooth = alpha_transition_smooth
        self.alpha_cross_memory = alpha_cross_memory
        self.alpha_cross_layer = alpha_cross_layer
        
        # Individual loss components
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
    
    def forward(self, 
                predictions: torch.Tensor,
                targets: torch.Tensor,
                states_history: Dict,
                memory_states: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute unified MESA-Net loss
        
        Args:
            predictions: Predicted precipitation (B, T, H, W)
            targets: Target precipitation (B, T, H, W)
            states_history: History of state evolution
            memory_states: Current memory states
            
        Returns:
            total_loss: Combined loss
            loss_components: Individual loss components for monitoring
        """
        # 1. Prediction loss (MSE + MAE)
        prediction_mse = self.mse_loss(predictions, targets)
        prediction_mae = self.mae_loss(predictions, targets)
        prediction_loss = prediction_mse + prediction_mae
        
        # 2. State entropy loss (prevent state collapse)
        state_entropy_loss = 0.0
        for memory_type in ['fast', 'slow', 'spatial', 'spatiotemporal']:
            state_probs = states_history['state_probs'][-1][memory_type]
            entropy = -torch.sum(state_probs * torch.log(state_probs + 1e-8), dim=1)
            state_entropy_loss += torch.mean(entropy)
        state_entropy_loss /= 4  # Average over memory types
        
        # 3. Transition smoothness loss
        transition_smooth_loss = 0.0
        if len(states_history['state_probs']) > 1:
            for memory_type in ['fast', 'slow', 'spatial', 'spatiotemporal']:
                current_probs = states_history['state_probs'][-1][memory_type]
                prev_probs = states_history['state_probs'][-2][memory_type]
                transition_smooth_loss += torch.mean(
                    torch.norm(current_probs - prev_probs, dim=1)
                )
            transition_smooth_loss /= 4
        
        # 4. Cross-memory coordination loss
        cross_memory_loss = 0.0
        memory_outputs = [
            memory_states['fast'],
            memory_states['slow'],
            memory_states['spatial'],
            memory_states['spatiotemporal']
        ]
        for i in range(len(memory_outputs)):
            for j in range(i + 1, len(memory_outputs)):
                correlation = torch.mean(
                    memory_outputs[i] * memory_outputs[j]
                )
                cross_memory_loss += torch.abs(correlation)
        
        # 5. Cross-layer consistency loss (PredRNN++ component)
        cross_layer_loss = 0.0
        if len(states_history['layer_outputs']) > 0:
            layer_outputs = states_history['layer_outputs'][-1]
            for i in range(len(layer_outputs) - 1):
                cross_layer_loss += self.mse_loss(
                    layer_outputs[i], layer_outputs[i + 1]
                )
        
        # Combine all losses
        total_loss = (
            self.alpha_prediction * prediction_loss +
            self.alpha_state_entropy * state_entropy_loss +
            self.alpha_transition_smooth * transition_smooth_loss +
            self.alpha_cross_memory * cross_memory_loss +
            self.alpha_cross_layer * cross_layer_loss
        )
        
        loss_components = {
            'prediction_loss': prediction_loss,
            'state_entropy_loss': state_entropy_loss,
            'transition_smooth_loss': transition_smooth_loss,
            'cross_memory_loss': cross_memory_loss,
            'cross_layer_loss': cross_layer_loss,
            'total_loss': total_loss
        }
        
        return total_loss, loss_components
