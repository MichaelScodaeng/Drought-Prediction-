# MINIMAL FIXES FOR mesanet.py
# Just add these small changes to your existing code

from typing import Dict, Tuple, List
import torch
import torch.nn as nn
from dataclasses import dataclass
from src.mesanet.state_machine import MemoryState, MemoryConfig
from src.mesanet.state_machine import MemoryStateMachine
from torch.nn import functional as F

class MESANetLayer(nn.Module):
    """Single MESA-Net layer with four memory types implementing the true MESA-Net spec"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 memory_config: MemoryConfig = None):
        super().__init__()
        
        if memory_config is None:
            memory_config = MemoryConfig()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Geographic conditioning following spec
        self.geo_embedding = nn.Sequential(
            nn.Linear(4, hidden_dim),  # lat, lon, elevation, land_sea_mask
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Four memory types implementing the expert team
        self.fast_memory = MemoryStateMachine("fast", memory_config, input_dim)
        self.slow_memory = MemoryStateMachine("slow", memory_config, input_dim)
        self.spatial_memory = MemoryStateMachine("spatial", memory_config, input_dim)
        self.spatiotemporal_memory = MemoryStateMachine("spatiotemporal", memory_config, input_dim)
        
        # Memory integration following spec
        self.memory_integration = nn.Sequential(
            nn.Conv2d(input_dim * 4, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, input_dim, 3, padding=1)
        )
        
        # Cross-layer memory (PredRNN++ component) - properly handle spatial dimensions
        self.cross_layer_projection = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.cross_layer_lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.cross_layer_back_projection = nn.Conv2d(hidden_dim, input_dim, 3, padding=1)
        
    def forward(self, 
                input_tensor: torch.Tensor,
                memory_states: Dict[str, torch.Tensor],
                state_probs: Dict[str, torch.Tensor],
                cross_layer_memory: torch.Tensor,
                geo_features: torch.Tensor) -> Tuple[torch.Tensor, Dict, Dict, torch.Tensor]:
        """
        Forward pass through MESA layer implementing the true MESA-Net framework
        """
        batch_size, channels, height, width = input_tensor.shape
        #print("Input tensor shape:", input_tensor.shape)
        if geo_features.shape[-2:] != (height, width):
            geo_features = F.interpolate(
                geo_features,
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )
            #print("Kuay")
        # Handle geo_features shape - ensure it's (B, 4, H, W)
        if geo_features.dim() == 3:  # (4, H, W)
            geo_features = geo_features.unsqueeze(0).expand(batch_size, -1, -1, -1)
        elif geo_features.dim() == 4 and geo_features.size(0) == 1:  # (1, 4, H, W)
            geo_features = geo_features.expand(batch_size, -1, -1, -1)
        
        # Geographic conditioning with proper reshaping
        geo_reshaped = geo_features.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        geo_embedding = self.geo_embedding(geo_reshaped)  # (B*H*W, input_dim)
        geo_embedding = geo_embedding.view(batch_size, height, width, -1).permute(0, 3, 1, 2)
        #print("geo embedding: ", geo_embedding.shape)
        # Apply geographic conditioning
        conditioned_input = input_tensor + geo_embedding
        
        # Create context for cross-memory interactions
        context = {
            'spatial_context': conditioned_input,
            'temporal_context': cross_layer_memory
        }
        
        # Process through each memory type with proper argument passing
        memory_outputs = {}
        updated_memory_states = {}
        updated_state_probs = {}
        
        # Fast memory (weather events) - The Weather Watcher
        fast_output, fast_state_probs = self.fast_memory(
            conditioned_input,
            memory_states['fast'],
            state_probs['fast'],
            context,
            memory_states,  # Pass all memory states for cross-memory attention
            geo_features
        )
        memory_outputs['fast'] = fast_output
        updated_memory_states['fast'] = fast_output
        updated_state_probs['fast'] = fast_state_probs
        
        # Slow memory (climate patterns) - The Climate Analyst
        slow_output, slow_state_probs = self.slow_memory(
            conditioned_input,
            memory_states['slow'],
            state_probs['slow'],
            context,
            memory_states,
            geo_features
        )
        memory_outputs['slow'] = slow_output
        updated_memory_states['slow'] = slow_output
        updated_state_probs['slow'] = slow_state_probs
        
        # Spatial memory (regional patterns) - The Regional Coordinator
        spatial_output, spatial_state_probs = self.spatial_memory(
            conditioned_input,
            memory_states['spatial'],
            state_probs['spatial'],
            context,
            memory_states,
            geo_features
        )
        memory_outputs['spatial'] = spatial_output
        updated_memory_states['spatial'] = spatial_output
        updated_state_probs['spatial'] = spatial_state_probs
        
        # Spatiotemporal memory (coordination) - The Integration Manager
        sptemp_output, sptemp_state_probs = self.spatiotemporal_memory(
            conditioned_input,
            memory_states['spatiotemporal'],
            state_probs['spatiotemporal'],
            context,
            memory_states,
            geo_features
        )
        memory_outputs['spatiotemporal'] = sptemp_output
        updated_memory_states['spatiotemporal'] = sptemp_output
        updated_state_probs['spatiotemporal'] = sptemp_state_probs
        
        # Integrate memory outputs from the four expert systems
        combined_memory = torch.cat([
            memory_outputs['fast'],
            memory_outputs['slow'],
            memory_outputs['spatial'],
            memory_outputs['spatiotemporal']
        ], dim=1)
        
        integrated_output = self.memory_integration(combined_memory)
        #print("integrated_output", integrated_output.shape)
        
        # Cross-layer memory update (PredRNN++) - proper spatial handling
        # Project to hidden dimension
        projected_output = self.cross_layer_projection(integrated_output)
        projected_cross_memory = self.cross_layer_projection(cross_layer_memory)
        #print("projected_output", projected_output.shape)
        #print("projected_cross_memory", projected_cross_memory.shape)
        
        # Average pool for LSTM processing
        pooled_output = F.adaptive_avg_pool2d(projected_output, (1, 1)).squeeze(-1).squeeze(-1)
        pooled_cross_memory = F.adaptive_avg_pool2d(projected_cross_memory, (1, 1)).squeeze(-1).squeeze(-1)
        #print("pooled_output", pooled_output.shape)
        #print("pooled_cross_memory", pooled_cross_memory.shape)
        
        # LSTM cell with proper hidden state management
        device = input_tensor.device
        hidden_state = torch.zeros(batch_size, self.hidden_dim, device=device)
        cell_state = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        new_hidden, new_cell = self.cross_layer_lstm(
            pooled_output, (hidden_state, cell_state)
        )
        
        # Project back to spatial dimensions
        cross_layer_features = new_hidden.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
        updated_cross_layer_memory = self.cross_layer_back_projection(cross_layer_features)
        #print("updated_cross_layer_memory", updated_cross_layer_memory.shape)

        
        # Residual connection for stability
        updated_cross_layer_memory = 0.9 * cross_layer_memory + 0.1 * updated_cross_layer_memory
        #print("updated_cross_layer_memory", updated_cross_layer_memory.shape)
        
        # 🔧 FIX #1: Small memory cleanup (ADD THIS LINE)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return (
            integrated_output,
            updated_memory_states,
            updated_state_probs,
            updated_cross_layer_memory
        )

class MESANet(nn.Module):
    """Complete MESA-Net architecture implementing the revolutionary adaptive framework"""
    
    def __init__(self,
                 input_channels: int,
                 num_layers: int = 3,
                 hidden_dim: int = 128,
                 memory_config: MemoryConfig = None):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_channels = input_channels
        
        # Input projection
        self.input_projection = nn.Conv2d(input_channels, hidden_dim, 3, padding=1)
        
        # MESA layers - the hierarchical expert team
        self.mesa_layers = nn.ModuleList([
            MESANetLayer(hidden_dim, hidden_dim, memory_config)
            for _ in range(num_layers)
        ])
        
        # Output head for precipitation prediction
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, 1, 3, padding=1),  # Precipitation output
            nn.ReLU()  # Non-negative precipitation
        )
        
        # For autoregressive forecasting - project precipitation back to hidden space
        self.forecast_projection = nn.Conv2d(1, hidden_dim, 3, padding=1)
        
    def forward(self, 
                input_sequence: torch.Tensor,
                geo_features: torch.Tensor,
                forecast_steps: int = 4) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through complete MESA-Net implementing the adaptive expert framework
        """
        batch_size, seq_len, channels, height, width = input_sequence.shape
        
        # Initialize the expert team's memory and state probabilities
        memory_states = self._initialize_memory_states(batch_size, height, width)
        state_probs = self._initialize_state_probs(batch_size)
        cross_layer_memories = self._initialize_cross_layer_memory(batch_size, height, width)
        #print("Input sequence shape:", input_sequence.shape)
        #print("Geo features shape:", geo_features.shape)
        #print("Memory states initialized:", memory_states.keys())
        #print("State probabilities initialized:", state_probs.keys())
        #print("Cross-layer memories initialized:", len(cross_layer_memories))
        
        # Track state evolution for interpretability analysis
        states_history = {
            'memory_states': [],
            'state_probs': [],
            'layer_outputs': []
        }
        
        # Process input sequence through the adaptive expert team
        last_layer_output = None
        for t in range(seq_len):
            current_input = input_sequence[:, t]
            projected_input = self.input_projection(current_input)
            
            layer_outputs = []
            layer_input = projected_input
            
            # Process through hierarchical MESA layers
            for layer_idx, mesa_layer in enumerate(self.mesa_layers):
                # 🔧 FIX #2: Add gradient checkpointing for memory efficiency (OPTIONAL)
                if self.training and hasattr(torch.utils, 'checkpoint'):
                    layer_output, memory_states, state_probs, cross_layer_memories[layer_idx] = \
                        torch.utils.checkpoint.checkpoint(
                            mesa_layer, layer_input, memory_states, state_probs, 
                            cross_layer_memories[layer_idx], geo_features,
                            preserve_rng_state=False
                        )
                else:
                    layer_output, memory_states, state_probs, cross_layer_memories[layer_idx] = mesa_layer(
                        layer_input,
                        memory_states,
                        state_probs,
                        cross_layer_memories[layer_idx],
                        geo_features
                    )
                layer_outputs.append(layer_output)
                layer_input = layer_output
            
            last_layer_output = layer_outputs[-1]
            
            # 🔧 FIX #3: More efficient state history tracking (REPLACE THESE LINES)
            # Store states for interpretability analysis (only every 3rd timestep to save memory)
            if t % 3 == 0 or t == seq_len - 1:  # Store every 3rd timestep + last timestep
                states_history['memory_states'].append({k: v.clone() for k, v in memory_states.items()})
                states_history['state_probs'].append({k: v.clone() for k, v in state_probs.items()})
                states_history['layer_outputs'].append([output.clone() for output in layer_outputs])
        
        # Generate autoregressive forecasts using learned adaptive behavior
        forecasts = []
        current_state = last_layer_output
        #print("current_state ", current_state.shape)
        
        for step in range(forecast_steps):
            # Generate precipitation prediction
            forecast_output = self.output_head(current_state)
            forecasts.append(forecast_output.squeeze(1))  # Remove channel dimension
            
            # Prepare for next autoregressive step
            # Project precipitation back to hidden dimensions for continued processing
            projected_forecast = self.forecast_projection(forecast_output)
            
            # Continue adaptive processing for next forecast step
            layer_input = projected_forecast
            for layer_idx, mesa_layer in enumerate(self.mesa_layers):
                # Use same checkpointing logic as above
                if self.training and hasattr(torch.utils, 'checkpoint'):
                    layer_output, memory_states, state_probs, cross_layer_memories[layer_idx] = \
                        torch.utils.checkpoint.checkpoint(
                            mesa_layer, layer_input, memory_states, state_probs, 
                            cross_layer_memories[layer_idx], geo_features,
                            preserve_rng_state=False
                        )
                else:
                    layer_output, memory_states, state_probs, cross_layer_memories[layer_idx] = mesa_layer(
                        layer_input,
                        memory_states,
                        state_probs,
                        cross_layer_memories[layer_idx],
                        geo_features
                    )
                layer_input = layer_output
            
            current_state = layer_output
            
            #print("current_state ", current_state.shape)
            
            # Store forecast step states for analysis (only for last forecast step)
            if step == forecast_steps - 1:
                states_history['memory_states'].append({k: v.clone() for k, v in memory_states.items()})
                states_history['state_probs'].append({k: v.clone() for k, v in state_probs.items()})
        
        forecast_tensor = torch.stack(forecasts, dim=1)  # (B, forecast_steps, H, W)
        #print("forecast_tensor ", forecast_tensor.shape)
        return forecast_tensor, states_history
    
    def _initialize_memory_states(self, batch_size: int, height: int, width: int) -> Dict[str, torch.Tensor]:
        """Initialize memory states for the four expert systems"""
        device = next(self.parameters()).device
        
        return {
            'fast': torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            'slow': torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            'spatial': torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            'spatiotemporal': torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        }
    
    def _initialize_state_probs(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Initialize state probabilities for the adaptive behaviors"""
        device = next(self.parameters()).device
        
        # Start with uniform distribution over states - let the system learn when to adapt
        uniform_probs = torch.ones(batch_size, 3, device=device) / 3.0
        
        return {
            'fast': uniform_probs.clone(),      # Weather Watcher states
            'slow': uniform_probs.clone(),      # Climate Analyst states  
            'spatial': uniform_probs.clone(),   # Regional Coordinator states
            'spatiotemporal': uniform_probs.clone()  # Integration Manager states
        }
    
    def _initialize_cross_layer_memory(self, batch_size: int, height: int, width: int) -> List[torch.Tensor]:
        """Initialize cross-layer memory for hierarchical processing (PredRNN++ component)"""
        device = next(self.parameters()).device
        
        return [
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
            for _ in range(self.num_layers)
        ]
    
    def get_state_interpretations(self, states_history: Dict) -> Dict[str, any]:
        """
        Analyze the learned state patterns for meteorological interpretability
        
        Returns insights into when and why the model changes its processing behavior
        """
        interpretations = {}
        
        memory_types = ['fast', 'slow', 'spatial', 'spatiotemporal']
        state_names = {
            'fast': ['Alert', 'Normal', 'Suppressed'],
            'slow': ['Accumulating', 'Stable', 'Adapting'], 
            'spatial': ['Local', 'Regional', 'Global'],
            'spatiotemporal': ['Synchronized', 'Leading', 'Following']
        }
        
        for memory_type in memory_types:
            state_evolution = []
            for timestep in states_history['state_probs']:
                if memory_type in timestep:
                    state_probs = timestep[memory_type]
                    avg_probs = torch.mean(state_probs, dim=0).cpu().numpy()
                    state_evolution.append(avg_probs)
            
            if state_evolution:
                state_evolution = torch.tensor(state_evolution)
                
                interpretations[f'{memory_type}_evolution'] = state_evolution
                interpretations[f'{memory_type}_dominant_states'] = torch.argmax(state_evolution, dim=1)
                interpretations[f'{memory_type}_state_names'] = state_names[memory_type]
                
                # Identify when the model changes processing modes
                state_changes = torch.diff(torch.argmax(state_evolution, dim=1))
                interpretations[f'{memory_type}_transition_points'] = torch.nonzero(state_changes).squeeze()
        
        return interpretations

# 🔧 SUMMARY OF MINIMAL CHANGES:
"""
WHAT WAS ADDED (3 tiny changes):

1. ONE LINE in MESANetLayer.forward():
   if torch.cuda.is_available():
       torch.cuda.empty_cache()

2. GRADIENT CHECKPOINTING (optional, helps with memory):
   if self.training and hasattr(torch.utils, 'checkpoint'):
       layer_output, ... = torch.utils.checkpoint.checkpoint(mesa_layer, ...)

3. EFFICIENT STATE TRACKING:
   if t % 3 == 0 or t == seq_len - 1:  # Only store every 3rd timestep
       states_history[...].append(...)

THAT'S IT! Your original code structure is 100% preserved.
"""