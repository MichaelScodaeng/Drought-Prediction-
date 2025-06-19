from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple
import torch
import torch.nn as nn

class MemoryState(Enum):
    """Enum for different memory states"""
    # Fast Memory States
    FAST_ALERT = "fast_alert"
    FAST_NORMAL = "fast_normal" 
    FAST_SUPPRESSED = "fast_suppressed"
    
    # Slow Memory States
    SLOW_ACCUMULATING = "slow_accumulating"
    SLOW_STABLE = "slow_stable"
    SLOW_ADAPTING = "slow_adapting"
    
    # Spatial Memory States
    SPATIAL_LOCAL = "spatial_local"
    SPATIAL_REGIONAL = "spatial_regional"
    SPATIAL_GLOBAL = "spatial_global"
    
    # Spatiotemporal Memory States
    SPTEMP_SYNCHRONIZED = "sptemp_synchronized"
    SPTEMP_LEADING = "sptemp_leading"
    SPTEMP_FOLLOWING = "sptemp_following"

@dataclass
class MemoryConfig:
    """Configuration for each memory type"""
    num_states: int = 3
    hidden_dim: int = 128
    learning_rates: Dict[str, float] = None
    spatial_kernels: Dict[str, int] = None
    
    def __post_init__(self):
        if self.learning_rates is None:
            self.learning_rates = {
                "alert": 0.1,
                "normal": 0.01,
                "suppressed": 0.001
            }
        
        if self.spatial_kernels is None:
            self.spatial_kernels = {
                "local": 3,
                "regional": 5,
                "global": 7
            }

class StateTransitionNetwork(nn.Module):
    """Attention-based state transition mechanism"""
    
    def __init__(self, 
                 input_dim: int,
                 num_states: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.num_states = num_states
        
        # Multi-head attention for context analysis
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            batch_first=True
        )
        
        # State transition network
        self.transition_net = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),  # spatial + temporal + current
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_states),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, 
                current_input: torch.Tensor,
                spatial_context: torch.Tensor,
                temporal_context: torch.Tensor,
                current_state: torch.Tensor) -> torch.Tensor:
        """
        Compute state transition probabilities
        
        Args:
            current_input: Current input features
            spatial_context: Spatial context features
            temporal_context: Temporal context features
            current_state: Current state probabilities
            
        Returns:
            next_state_probs: Softmax probabilities over states
        """
        # Attention-based context computation
        spatial_attn, _ = self.spatial_attention(
            current_input, spatial_context, spatial_context
        )
        
        temporal_attn, _ = self.temporal_attention(
            current_input, temporal_context, temporal_context
        )
        
        # Combine contexts
        combined_context = torch.cat([
            current_input.mean(dim=1),  # Pool spatial dimensions
            spatial_attn.mean(dim=1),
            temporal_attn.mean(dim=1)
        ], dim=-1)
        
        # Compute state transition
        next_state_probs = self.transition_net(combined_context)
        
        return next_state_probs

class MemoryStateMachine(nn.Module):
    """Individual memory type with state-dependent processing"""
    
    def __init__(self, 
                 memory_type: str,
                 config: MemoryConfig,
                 input_dim: int):
        super().__init__()
        self.memory_type = memory_type
        self.config = config
        self.num_states = config.num_states
        
        # State transition mechanism
        self.state_transition = StateTransitionNetwork(
            input_dim=input_dim,
            num_states=self.num_states
        )
        
        # State-dependent processing networks
        self.state_processors = nn.ModuleDict()
        for state_idx in range(self.num_states):
            self.state_processors[f"state_{state_idx}"] = self._create_state_processor(
                state_idx, input_dim
            )
        
        # Memory update networks
        self.memory_update = nn.LSTMCell(input_dim, config.hidden_dim)
        
    def _create_state_processor(self, state_idx: int, input_dim: int) -> nn.Module:
        """Create state-specific processing network"""
        if self.memory_type == "fast":
            # Fast memory: Different sensitivity levels
            if state_idx == 0:  # Alert state
                return nn.Sequential(
                    nn.Conv2d(input_dim, input_dim, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(input_dim, input_dim, 3, padding=1)
                )
            else:  # Normal/Suppressed states
                return nn.Sequential(
                    nn.Conv2d(input_dim, input_dim, 3, padding=1),
                    nn.ReLU()
                )
        
        elif self.memory_type == "spatial":
            # Spatial memory: Different receptive fields
            kernel_sizes = [3, 5, 7]  # Local, Regional, Global
            kernel_size = kernel_sizes[state_idx]
            padding = kernel_size // 2
            
            return nn.Sequential(
                nn.Conv2d(input_dim, input_dim, kernel_size, padding=padding),
                nn.ReLU(),
                nn.Conv2d(input_dim, input_dim, kernel_size, padding=padding)
            )
        
        else:
            # Default processor
            return nn.Sequential(
                nn.Conv2d(input_dim, input_dim, 3, padding=1),
                nn.ReLU()
            )
    
    def forward(self, 
                input_tensor: torch.Tensor,
                memory_state: torch.Tensor,
                current_state_probs: torch.Tensor,
                context: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through memory state machine
        
        Args:
            input_tensor: Current input
            memory_state: Current memory state
            current_state_probs: Current state probabilities
            context: Context from other memories/inputs
            
        Returns:
            updated_memory: Updated memory state
            new_state_probs: New state probabilities
        """
        # Compute state transitions
        new_state_probs = self.state_transition(
            input_tensor,
            context.get('spatial_context', input_tensor),
            context.get('temporal_context', input_tensor),
            current_state_probs
        )
        
        # State-dependent processing
        processed_outputs = []
        for state_idx in range(self.num_states):
            state_processor = self.state_processors[f"state_{state_idx}"]
            state_output = state_processor(input_tensor)
            processed_outputs.append(state_output)
        
        # Weighted combination based on state probabilities
        combined_output = torch.zeros_like(processed_outputs[0])
        for state_idx, output in enumerate(processed_outputs):
            weight = new_state_probs[:, state_idx:state_idx+1, None, None]
            combined_output += weight * output
        
        # Update memory
        batch_size = input_tensor.size(0)
        flat_input = combined_output.view(batch_size, -1)
        flat_memory = memory_state.view(batch_size, -1)
        
        new_memory, _ = self.memory_update(flat_input, flat_memory)
        updated_memory = new_memory.view_as(memory_state)
        
        return updated_memory, new_state_probs