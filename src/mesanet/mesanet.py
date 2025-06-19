from typing import Dict, Tuple, List
import torch
import torch.nn as nn
from dataclasses import dataclass
from src.mesanet.state_machine import MemoryState, MemoryConfig
from src.mesanet.state_machine import MemoryStateMachine
from torch.nn import functional as F

class MESANetLayer(nn.Module):
    """Single MESA-Net layer with four memory types"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 memory_config: MemoryConfig = None):
        super().__init__()
        
        if memory_config is None:
            memory_config = MemoryConfig()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Geographic conditioning
        self.geo_embedding = nn.Sequential(
            nn.Linear(4, hidden_dim),  # lat, lon, elevation, land_sea_mask
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Four memory types
        self.fast_memory = MemoryStateMachine("fast", memory_config, input_dim)
        self.slow_memory = MemoryStateMachine("slow", memory_config, input_dim)
        self.spatial_memory = MemoryStateMachine("spatial", memory_config, input_dim)
        self.spatiotemporal_memory = MemoryStateMachine("spatiotemporal", memory_config, input_dim)
        
        # Cross-memory interaction
        self.cross_memory_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Memory integration
        self.memory_integration = nn.Sequential(
            nn.Conv2d(input_dim * 4, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, input_dim, 3, padding=1)
        )
        
        # Cross-layer memory (PredRNN++ component)
        self.cross_layer_lstm = nn.LSTMCell(input_dim, hidden_dim)
        
    def forward(self, 
                input_tensor: torch.Tensor,
                memory_states: Dict[str, torch.Tensor],
                state_probs: Dict[str, torch.Tensor],
                cross_layer_memory: torch.Tensor,
                geo_features: torch.Tensor) -> Tuple[torch.Tensor, Dict, Dict, torch.Tensor]:
        """
        Forward pass through MESA layer
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            memory_states: Dictionary of memory states for each type
            state_probs: Dictionary of state probabilities
            cross_layer_memory: Cross-layer memory from PredRNN++
            geo_features: Geographic features
            
        Returns:
            output: Layer output
            updated_memory_states: Updated memory states
            updated_state_probs: Updated state probabilities
            updated_cross_layer_memory: Updated cross-layer memory
        """
        # Geographic conditioning
        geo_embedding = self.geo_embedding(geo_features)
        conditioned_input = input_tensor + geo_embedding.unsqueeze(-1).unsqueeze(-1)
        
        # Process through each memory type
        memory_outputs = {}
        updated_memory_states = {}
        updated_state_probs = {}
        
        # Create context for cross-memory interactions
        context = {
            'spatial_context': conditioned_input,
            'temporal_context': cross_layer_memory
        }
        
        # Fast memory (weather events)
        fast_output, fast_states = self.fast_memory(
            conditioned_input,
            memory_states['fast'],
            state_probs['fast'],
            context
        )
        memory_outputs['fast'] = fast_output
        updated_memory_states['fast'] = fast_states
        
        # Slow memory (climate patterns)
        slow_output, slow_states = self.slow_memory(
            conditioned_input,
            memory_states['slow'],
            state_probs['slow'],
            context
        )
        memory_outputs['slow'] = slow_output
        updated_memory_states['slow'] = slow_states
        
        # Spatial memory (regional patterns)
        spatial_output, spatial_states = self.spatial_memory(
            conditioned_input,
            memory_states['spatial'],
            state_probs['spatial'],
            context
        )
        memory_outputs['spatial'] = spatial_output
        updated_memory_states['spatial'] = spatial_states
        
        # Spatiotemporal memory (coordination)
        sptemp_output, sptemp_states = self.spatiotemporal_memory(
            conditioned_input,
            memory_states['spatiotemporal'],
            state_probs['spatiotemporal'],
            context
        )
        memory_outputs['spatiotemporal'] = sptemp_output
        updated_memory_states['spatiotemporal'] = sptemp_states
        
        # Integrate memory outputs
        combined_memory = torch.cat([
            memory_outputs['fast'],
            memory_outputs['slow'],
            memory_outputs['spatial'],
            memory_outputs['spatiotemporal']
        ], dim=1)
        
        integrated_output = self.memory_integration(combined_memory)
        
        # Cross-layer memory update (PredRNN++)
        batch_size = input_tensor.size(0)
        flat_output = integrated_output.view(batch_size, -1)
        flat_cross_memory = cross_layer_memory.view(batch_size, -1)
        
        updated_cross_layer_memory, _ = self.cross_layer_lstm(
            flat_output, flat_cross_memory
        )
        updated_cross_layer_memory = updated_cross_layer_memory.view_as(cross_layer_memory)
        
        return (
            integrated_output,
            updated_memory_states,
            updated_state_probs,
            updated_cross_layer_memory
        )

class MESANet(nn.Module):
    """Complete MESA-Net architecture"""
    
    def __init__(self,
                 input_channels: int,
                 num_layers: int = 3,
                 hidden_dim: int = 128,
                 memory_config: MemoryConfig = None):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Conv2d(input_channels, hidden_dim, 3, padding=1)
        
        # MESA layers
        self.mesa_layers = nn.ModuleList([
            MESANetLayer(hidden_dim, hidden_dim, memory_config)
            for _ in range(num_layers)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, 1, 3, padding=1),  # Precipitation output
            nn.ReLU()  # Non-negative precipitation
        )
        
    def forward(self, 
                input_sequence: torch.Tensor,
                geo_features: torch.Tensor,
                forecast_steps: int = 4) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through complete MESA-Net
        
        Args:
            input_sequence: (B, T, C, H, W) - Input meteorological sequence
            geo_features: (B, 4, H, W) - Geographic features
            forecast_steps: Number of forecast time steps
            
        Returns:
            forecast: (B, forecast_steps, H, W) - Precipitation forecast
            states_history: Dictionary containing state evolution for analysis
        """
        batch_size, seq_len, channels, height, width = input_sequence.shape
        
        # Initialize memory states and state probabilities
        memory_states = self._initialize_memory_states(batch_size, height, width)
        state_probs = self._initialize_state_probs(batch_size)
        cross_layer_memories = self._initialize_cross_layer_memory(batch_size, height, width)
        
        states_history = {
            'memory_states': [],
            'state_probs': [],
            'layer_outputs': []
        }
        
        # Process input sequence
        for t in range(seq_len):
            current_input = input_sequence[:, t]
            projected_input = self.input_projection(current_input)
            
            layer_outputs = []
            layer_input = projected_input
            
            # Process through MESA layers
            for layer_idx, mesa_layer in enumerate(self.mesa_layers):
                layer_output, memory_states, state_probs, cross_layer_memories[layer_idx] = mesa_layer(
                    layer_input,
                    memory_states,
                    state_probs,
                    cross_layer_memories[layer_idx],
                    geo_features
                )
                layer_outputs.append(layer_output)
                layer_input = layer_output
            
            # Store states for analysis
            states_history['memory_states'].append(memory_states.copy())
            states_history['state_probs'].append(state_probs.copy())
            states_history['layer_outputs'].append(layer_outputs.copy())
        
        # Generate forecasts
        forecasts = []
        current_state = layer_outputs[-1]  # Last layer output
        
        for step in range(forecast_steps):
            # Generate next time step
            forecast_output = self.output_head(current_state)
            forecasts.append(forecast_output)
            
            # Update state for next prediction (autoregressive)
            # Use forecast as input for next step
            projected_forecast = self.input_projection(
                torch.cat([forecast_output, current_state[:, 1:]], dim=1)
            )
            
            layer_input = projected_forecast
            for layer_idx, mesa_layer in enumerate(self.mesa_layers):
                layer_output, memory_states, state_probs, cross_layer_memories[layer_idx] = mesa_layer(
                    layer_input,
                    memory_states,
                    state_probs,
                    cross_layer_memories[layer_idx],
                    geo_features
                )
                layer_input = layer_output
            
            current_state = layer_output
        
        forecast_tensor = torch.stack(forecasts, dim=1)
        
        return forecast_tensor, states_history
    
    def _initialize_memory_states(self, batch_size: int, height: int, width: int) -> Dict[str, torch.Tensor]:
        """Initialize memory states for all memory types"""
        device = next(self.parameters()).device
        
        return {
            'fast': torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            'slow': torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            'spatial': torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            'spatiotemporal': torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        }
    
    def _initialize_state_probs(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Initialize state probabilities for all memory types"""
        device = next(self.parameters()).device
        
        # Start with uniform distribution over states
        uniform_probs = torch.ones(batch_size, 3, device=device) / 3.0
        
        return {
            'fast': uniform_probs.clone(),
            'slow': uniform_probs.clone(),
            'spatial': uniform_probs.clone(),
            'spatiotemporal': uniform_probs.clone()
        }
    
    def _initialize_cross_layer_memory(self, batch_size: int, height: int, width: int) -> List[torch.Tensor]:
        """Initialize cross-layer memory for PredRNN++ component"""
        device = next(self.parameters()).device
        
        return [
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
            for _ in range(self.num_layers)
        ]

# =============================================================================
# 4. LOSS FUNCTIONS MODULE
# =============================================================================
