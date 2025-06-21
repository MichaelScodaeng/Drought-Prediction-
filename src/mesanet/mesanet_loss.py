from typing import Dict, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass

class MESANetLoss(nn.Module):
    """Unified loss function for MESA-Net with robust error handling"""
    
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
        Compute unified MESA-Net loss with robust error handling
        
        Args:
            predictions: Predicted precipitation (B, T, H, W)
            targets: Target precipitation (B, T, H, W)
            states_history: History of state evolution
            memory_states: Current memory states
            
        Returns:
            total_loss: Combined loss
            loss_components: Individual loss components for monitoring
        """
        device = predictions.device
        
        # 1. Prediction loss (MSE + MAE) - Always works
        prediction_mse = self.mse_loss(predictions, targets)
        prediction_mae = self.mae_loss(predictions, targets)
        prediction_loss = prediction_mse + prediction_mae
        
        # 2. State entropy loss (prevent state collapse) - WITH ERROR HANDLING
        state_entropy_loss = torch.tensor(0.0, device=device)
        
        try:
            if (states_history and 'state_probs' in states_history and 
                len(states_history['state_probs']) > 0):
                
                latest_state_probs = states_history['state_probs'][-1]
                entropy_count = 0
                
                for memory_type in ['fast', 'slow', 'spatial', 'spatiotemporal']:
                    if memory_type in latest_state_probs:
                        state_probs = latest_state_probs[memory_type]
                        
                        # 🔧 FIX: Safe entropy computation
                        if state_probs.numel() > 0:
                            # Clamp probabilities to prevent log(0)
                            state_probs_safe = torch.clamp(state_probs, min=1e-8, max=1.0)
                            entropy = -torch.sum(state_probs_safe * torch.log(state_probs_safe), dim=1)
                            state_entropy_loss += torch.mean(entropy)
                            entropy_count += 1
                
                if entropy_count > 0:
                    state_entropy_loss /= entropy_count  # Average over available memory types
                    
        except Exception as e:
            # If state entropy computation fails, use zero (don't crash training)
            state_entropy_loss = torch.tensor(0.0, device=device)
        
        # 3. Transition smoothness loss - WITH ERROR HANDLING
        transition_smooth_loss = torch.tensor(0.0, device=device)
        
        try:
            if (states_history and 'state_probs' in states_history and 
                len(states_history['state_probs']) > 1):
                
                current_state_probs = states_history['state_probs'][-1]
                prev_state_probs = states_history['state_probs'][-2]
                transition_count = 0
                
                for memory_type in ['fast', 'slow', 'spatial', 'spatiotemporal']:
                    if (memory_type in current_state_probs and 
                        memory_type in prev_state_probs):
                        
                        current_probs = current_state_probs[memory_type]
                        prev_probs = prev_state_probs[memory_type]
                        
                        # 🔧 FIX: Safe norm computation
                        if current_probs.shape == prev_probs.shape and current_probs.numel() > 0:
                            transition_smooth_loss += torch.mean(
                                torch.norm(current_probs - prev_probs, dim=1)
                            )
                            transition_count += 1
                
                if transition_count > 0:
                    transition_smooth_loss /= transition_count
                    
        except Exception as e:
            # If transition smoothness computation fails, use zero
            transition_smooth_loss = torch.tensor(0.0, device=device)
        
        # 4. Cross-memory coordination loss - WITH ERROR HANDLING
        cross_memory_loss = torch.tensor(0.0, device=device)
        
        try:
            # 🔧 FIX: Safe memory states access
            expected_keys = ['fast', 'slow', 'spatial', 'spatiotemporal']
            available_memories = []
            
            for key in expected_keys:
                if key in memory_states and memory_states[key] is not None:
                    # Check if tensor is not empty
                    if memory_states[key].numel() > 0:
                        available_memories.append(memory_states[key])
            
            # Only compute cross-memory loss if we have multiple memories
            if len(available_memories) >= 2:
                correlation_count = 0
                for i in range(len(available_memories)):
                    for j in range(i + 1, len(available_memories)):
                        mem_i = available_memories[i]
                        mem_j = available_memories[j]
                        
                        # Ensure tensors have same shape
                        if mem_i.shape == mem_j.shape:
                            correlation = torch.mean(mem_i * mem_j)
                            cross_memory_loss += torch.abs(correlation)
                            correlation_count += 1
                
                if correlation_count > 0:
                    cross_memory_loss /= correlation_count
                    
        except Exception as e:
            # If cross-memory computation fails, use zero
            cross_memory_loss = torch.tensor(0.0, device=device)
        
        # 5. Cross-layer consistency loss (PredRNN++ component) - WITH ERROR HANDLING
        cross_layer_loss = torch.tensor(0.0, device=device)
        
        try:
            if (states_history and 'layer_outputs' in states_history and 
                len(states_history['layer_outputs']) > 0):
                
                layer_outputs = states_history['layer_outputs'][-1]
                
                # 🔧 FIX: Safe layer outputs access
                if isinstance(layer_outputs, list) and len(layer_outputs) >= 2:
                    layer_count = 0
                    for i in range(len(layer_outputs) - 1):
                        output_i = layer_outputs[i]
                        output_j = layer_outputs[i + 1]
                        
                        # Check if tensors are valid and same shape
                        if (output_i is not None and output_j is not None and 
                            output_i.shape == output_j.shape and output_i.numel() > 0):
                            cross_layer_loss += self.mse_loss(output_i, output_j)
                            layer_count += 1
                    
                    if layer_count > 0:
                        cross_layer_loss /= layer_count
                        
        except Exception as e:
            # If cross-layer computation fails, use zero
            cross_layer_loss = torch.tensor(0.0, device=device)
        
        # 6. Combine all losses with safety checks
        total_loss = self.alpha_prediction * prediction_loss
        
        # Only add regularization losses if they're valid (not NaN/inf)
        if torch.isfinite(state_entropy_loss):
            total_loss += self.alpha_state_entropy * state_entropy_loss
            
        if torch.isfinite(transition_smooth_loss):
            total_loss += self.alpha_transition_smooth * transition_smooth_loss
            
        if torch.isfinite(cross_memory_loss):
            total_loss += self.alpha_cross_memory * cross_memory_loss
            
        if torch.isfinite(cross_layer_loss):
            total_loss += self.alpha_cross_layer * cross_layer_loss
        
        # 🔧 FIX: Ensure all loss components are tensors on correct device
        loss_components = {
            'prediction_loss': prediction_loss.detach(),
            'state_entropy_loss': state_entropy_loss.detach(),
            'transition_smooth_loss': transition_smooth_loss.detach(),
            'cross_memory_loss': cross_memory_loss.detach(),
            'cross_layer_loss': cross_layer_loss.detach(),
            'total_loss': total_loss.detach()
        }
        
        return total_loss, loss_components
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Return current loss component weights for analysis"""
        return {
            'alpha_prediction': self.alpha_prediction,
            'alpha_state_entropy': self.alpha_state_entropy,
            'alpha_transition_smooth': self.alpha_transition_smooth,
            'alpha_cross_memory': self.alpha_cross_memory,
            'alpha_cross_layer': self.alpha_cross_layer
        }
    
    def update_loss_weights(self, **kwargs):
        """Update loss component weights during training if needed"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"Updated {key} to {value}")

# 🔧 SIMPLE FALLBACK LOSS (if you want a minimal version for testing)
class SimpleMESANetLoss(nn.Module):
    """Simplified MESA-Net loss for testing/debugging"""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
    
    def forward(self, predictions, targets, states_history, memory_states):
        """Simple prediction loss only"""
        prediction_loss = self.mse_loss(predictions, targets) + 0.1 * self.mae_loss(predictions, targets)
        
        # Simple state entropy to prevent collapse
        state_entropy_loss = torch.tensor(0.0, device=predictions.device)
        if (states_history and 'state_probs' in states_history and 
            len(states_history['state_probs']) > 0):
            try:
                latest_states = states_history['state_probs'][-1]
                for memory_type, probs in latest_states.items():
                    if probs.numel() > 0:
                        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                        state_entropy_loss += torch.mean(entropy)
            except:
                pass
        
        total_loss = prediction_loss + 0.01 * state_entropy_loss
        
        return total_loss, {
            'prediction_loss': prediction_loss,
            'state_entropy_loss': state_entropy_loss,
            'transition_smooth_loss': torch.tensor(0.0, device=predictions.device),
            'cross_memory_loss': torch.tensor(0.0, device=predictions.device),
            'cross_layer_loss': torch.tensor(0.0, device=predictions.device),
            'total_loss': total_loss
        }