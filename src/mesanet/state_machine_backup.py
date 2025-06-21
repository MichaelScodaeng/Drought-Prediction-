from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

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
                "alert": 0.1,        # High sensitivity
                "normal": 0.01,      # Standard processing
                "suppressed": 0.001  # Low sensitivity
            }
        
        if self.spatial_kernels is None:
            self.spatial_kernels = {
                "local": 3,      # Immediate neighbors
                "regional": 5,   # State/province scale
                "global": 7      # Continental patterns
            }

class SpatialAttention(nn.Module):
    """Enhanced spatial attention for different states"""
    
    def __init__(self, input_dim: int, geo_dim: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.geo_projection = nn.Conv2d(geo_dim, input_dim, 1)
        self.attention_conv = nn.Conv2d(input_dim * 2, 1, 1)
        
    def forward(self, features: torch.Tensor, geo_features: torch.Tensor, 
                focus_mode: str = 'normal') -> torch.Tensor:
        """
        Compute spatial attention based on geographic context and focus mode
        
        Args:
            features: Input features (B, C, H, W)
            geo_features: Geographic features (B, 4, H, W)
            focus_mode: 'normal', 'extreme_gradients', 'global_patterns'
        """
        # Project geographic features to match input dimensions
        geo_proj = self.geo_projection(geo_features)
        
        # Combine features and geography
        combined = torch.cat([features, geo_proj], dim=1)
        
        # Compute attention weights
        attention_weights = torch.sigmoid(self.attention_conv(combined))
        
        # Apply focus-specific processing
        if focus_mode == 'extreme_gradients':
            # Enhance attention on regions with high gradients (Alert state)
            grad_x = torch.abs(features[:, :, :, 1:] - features[:, :, :, :-1])
            grad_y = torch.abs(features[:, :, 1:, :] - features[:, :, :-1, :])
            
            # Pad gradients to match original size
            grad_x = F.pad(grad_x, (0, 1, 0, 0))
            grad_y = F.pad(grad_y, (0, 0, 0, 1))
            
            gradient_magnitude = grad_x.mean(dim=1, keepdim=True) + grad_y.mean(dim=1, keepdim=True)
            attention_weights = attention_weights * (1 + gradient_magnitude)
            
        elif focus_mode == 'global_patterns':
            # Use global average pooling for global patterns
            global_avg = F.adaptive_avg_pool2d(features, (1, 1))
            attention_weights = attention_weights * global_avg.expand_as(attention_weights)
        
        return features * attention_weights

class CrossMemoryAttention(nn.Module):
    """Cross-memory attention mechanism"""
    
    def __init__(self, input_dim: int, num_memory_types: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.num_memory_types = num_memory_types
        self.output_projection = nn.Conv2d(input_dim // 4, input_dim, 1)
        self.memory_projection = nn.ModuleDict({
            'fast': nn.Conv2d(input_dim, input_dim // 4, 1),
            'slow': nn.Conv2d(input_dim, input_dim // 4, 1),
            'spatial': nn.Conv2d(input_dim, input_dim // 4, 1),
            'spatiotemporal': nn.Conv2d(input_dim, input_dim // 4, 1)
        })
        
        self.attention_weights = nn.Conv2d((input_dim // 4) * (num_memory_types - 1), num_memory_types - 1, 1)

        
    def forward(self, memory_states: Dict[str, torch.Tensor], 
                current_memory_type: str) -> torch.Tensor:
        """
        Compute cross-memory attention context
        
        Args:
            memory_states: Dictionary of memory states from other memory types
            current_memory_type: Current memory type to exclude from attention
            
        Returns:
            cross_memory_context: Attended context from other memories
        """
        # Project each memory type to smaller dimension
        projected_memories = []
        memory_types = []
        
        for mem_type, memory_state in memory_states.items():
            if mem_type != current_memory_type and mem_type in self.memory_projection:
                projected = self.memory_projection[mem_type](memory_state)
                projected_memories.append(projected)
                memory_types.append(mem_type)
        
        if not projected_memories:
            # Return zeros if no other memories available
            batch_size, _, height, width = list(memory_states.values())[0].shape
            device = list(memory_states.values())[0].device
            return torch.zeros(batch_size, self.input_dim, height, width, device=device)
        
        # Concatenate projected memories
        combined_memories = torch.cat(projected_memories, dim=1)  # (B, C_total, H, W)
        
        # Compute attention weights for each memory type
        attention_weights = F.softmax(
            self.attention_weights(combined_memories)[:, :len(projected_memories)], 
            dim=1
        )
        
        # Apply attention to get weighted combination
        attended_context = torch.zeros_like(projected_memories[0])
        for i, proj_mem in enumerate(projected_memories):
            weight = attention_weights[:, i:i+1]  # (B, 1, H, W)
            attended_context += weight * proj_mem
        
        # Project back to full dimensions
        context = self.output_projection(attended_context)
        
        return context

class StateTransitionNetwork(nn.Module):
    """Attention-based state transition mechanism implementing the MESA-Net spec"""
    
    def __init__(self, input_dim: int, num_states: int, hidden_dim: int = 64):
        super().__init__()
        self.num_states = num_states
        self.input_dim = input_dim
        
        # Spatial attention component
        self.spatial_attention = SpatialAttention(input_dim)
        
        # Cross-memory attention component  
        self.cross_memory_attention = CrossMemoryAttention(input_dim)
        
        # Context integration network
        self.context_integration = nn.Sequential(
            nn.Conv2d(input_dim * 3, hidden_dim, 3, padding=1),  # spatial + temporal + cross_memory
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_states),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, 
                current_input: torch.Tensor,
                spatial_context: torch.Tensor,
                temporal_context: torch.Tensor,
                current_state: torch.Tensor,
                memory_states: Dict[str, torch.Tensor],
                memory_type: str,
                geo_features: torch.Tensor) -> torch.Tensor:
        """
        Compute state transition probabilities following MESA-Net mathematical framework
        
        Args:
            current_input: Current input features (B, C, H, W)
            spatial_context: Spatial context features (B, C, H, W)
            temporal_context: Temporal context features (B, C, H, W)
            current_state: Current state probabilities (B, num_states)
            memory_states: States from all memory types
            memory_type: Current memory type
            geo_features: Geographic features (B, 4, H, W)
            
        Returns:
            next_state_probs: State transition probabilities (B, num_states)
        """
        # Spatial attention with geographic conditioning
        spatial_features = self.spatial_attention(spatial_context, geo_features)
        
        # Cross-memory attention
        cross_memory_features = self.cross_memory_attention(memory_states, memory_type)
        
        # Combine all context features
        combined_context = torch.cat([
            spatial_features,
            temporal_context,
            cross_memory_features
        ], dim=1)
        
        # Compute state transition probabilities
        next_state_probs = self.context_integration(combined_context)
        
        return next_state_probs

class MemoryStateMachine(nn.Module):
    """Individual memory type with true state-dependent processing following MESA-Net spec"""
    
    def __init__(self, memory_type: str, config: MemoryConfig, input_dim: int):
        super().__init__()
        self.memory_type = memory_type
        self.config = config
        self.num_states = config.num_states
        self.input_dim = input_dim
        
        # State transition mechanism
        self.state_transition = StateTransitionNetwork(
            input_dim=input_dim,
            num_states=self.num_states
        )
        
        # State-dependent processing networks with proper behaviors
        self.state_processors = nn.ModuleDict()
        for state_idx in range(self.num_states):
            self.state_processors[f"state_{state_idx}"] = self._create_state_processor(state_idx, input_dim)
        
        # Spatial memory preservation instead of flattening
        self.memory_integration = nn.Conv2d(input_dim, config.hidden_dim, 3, padding=1)
        self.memory_projection = nn.Conv2d(config.hidden_dim, input_dim, 3, padding=1)
        
    def _create_state_processor(self, state_idx: int, input_dim: int) -> nn.Module:
        """Create state-specific processing networks implementing MESA-Net behaviors"""
        
        if self.memory_type == "fast":
            if state_idx == 0:  # Alert state: High sensitivity, enhanced spatial attention
                return nn.ModuleDict({
                    'processor': nn.Sequential(
                        nn.Conv2d(input_dim, input_dim, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(input_dim, input_dim, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(input_dim, input_dim, 1)  # Extra processing for high sensitivity
                    ),
                    'attention': SpatialAttention(input_dim)
                })
            elif state_idx == 1:  # Normal state: Standard processing
                return nn.ModuleDict({
                    'processor': nn.Sequential(
                        nn.Conv2d(input_dim, input_dim, 3, padding=1),
                        nn.ReLU()
                    ),
                    'attention': SpatialAttention(input_dim)
                })
            else:  # Suppressed state: Minimal updates
                return nn.ModuleDict({
                    'processor': nn.Sequential(
                        nn.Conv2d(input_dim, input_dim, 3, padding=1)
                    ),
                    'attention': SpatialAttention(input_dim)
                })
        
        elif self.memory_type == "spatial":
            # Different receptive fields: Local(3x3), Regional(5x5), Global(7x7 + attention)
            kernel_sizes = [3, 5, 7]
            kernel_size = kernel_sizes[state_idx]
            padding = kernel_size // 2
            
            if state_idx == 2:  # Global state: Large receptive field + global attention
                return nn.ModuleDict({
                    'processor': nn.Sequential(
                        nn.Conv2d(input_dim, input_dim, kernel_size, padding=padding),
                        nn.ReLU(),
                        nn.Conv2d(input_dim, input_dim, kernel_size, padding=padding)
                    ),
                    'global_attention': nn.MultiheadAttention(input_dim, 8, batch_first=True)
                })
            else:
                return nn.ModuleDict({
                    'processor': nn.Sequential(
                        nn.Conv2d(input_dim, input_dim, kernel_size, padding=padding),
                        nn.ReLU(),
                        nn.Conv2d(input_dim, input_dim, kernel_size, padding=padding)
                    )
                })
        
        elif self.memory_type == "slow":
            # Different retention strengths: Accumulating, Stable, Adapting
            if state_idx == 0:  # Accumulating: Strong retention
                return nn.ModuleDict({
                    'processor': nn.Sequential(
                        nn.Conv2d(input_dim, input_dim, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(input_dim, input_dim, 3, padding=1),
                        nn.Sigmoid()  # Strong retention through sigmoid gating
                    )
                })
            elif state_idx == 1:  # Stable: Maintain patterns
                return nn.ModuleDict({
                    'processor': nn.Sequential(
                        nn.Conv2d(input_dim, input_dim, 3, padding=1),
                        nn.ReLU()
                    )
                })
            else:  # Adapting: Reduced retention for regime shifts
                return nn.ModuleDict({
                    'processor': nn.Sequential(
                        nn.Conv2d(input_dim, input_dim, 3, padding=1),
                        nn.ReLU(),
                        nn.Dropout2d(0.1)  # Reduced retention through dropout
                    )
                })
        
        else:  # spatiotemporal: Coordination states
            return nn.ModuleDict({
                'processor': nn.Sequential(
                    nn.Conv2d(input_dim, input_dim, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(input_dim, input_dim, 3, padding=1)
                )
            })
    
    def forward(self, 
                input_tensor: torch.Tensor,
                memory_state: torch.Tensor,
                current_state_probs: torch.Tensor,
                context: Dict[str, torch.Tensor],
                memory_states: Dict[str, torch.Tensor],
                geo_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass implementing the MESA-Net mathematical framework:
        M_m^{t+1} = Σᵢ State_weights_m^{t+1}[i] × P_m^{state_i}(X_t, M_m^t, α_m^{state_i})
        """
        # Compute state transitions using full context
        new_state_probs = self.state_transition(
            input_tensor,
            context.get('spatial_context', input_tensor),
            context.get('temporal_context', input_tensor),
            current_state_probs,
            memory_states,
            self.memory_type,
            geo_features
        )
        
        # State-dependent processing with different behaviors
        processed_outputs = []
        learning_rates = []
        
        for state_idx in range(self.num_states):
            state_processor_dict = self.state_processors[f"state_{state_idx}"]
            
            # Apply state-specific processing
            if self.memory_type == "fast":
                # Get state-dependent learning rate
                state_names = ["alert", "normal", "suppressed"]
                alpha = self.config.learning_rates[state_names[state_idx]]
                learning_rates.append(alpha)
                
                # Process with state-specific behavior
                processed = state_processor_dict['processor'](input_tensor)
                
                if state_idx == 0:  # Alert state: enhanced spatial attention
                    processed = state_processor_dict['attention'](
                        processed, geo_features, focus_mode='extreme_gradients'
                    )
                else:
                    processed = state_processor_dict['attention'](
                        processed, geo_features, focus_mode='normal'
                    )
                    
            elif self.memory_type == "spatial" and state_idx == 2:  # Global state
                processed = state_processor_dict['processor'](input_tensor)
                
                # Apply global attention for global spatial state
                B, C, H, W = processed.shape
                '''processed_flat = processed.view(B, C, H*W).permute(0, 2, 1)  # (B, H*W, C)
                
                global_attended, _ = state_processor_dict['global_attention'](
                    processed_flat, processed_flat, processed_flat
                )
                processed = global_attended.permute(0, 2, 1).view(B, C, H, W)'''
                # Downsample spatially to reduce sequence length before attention
                pooled = F.adaptive_avg_pool2d(processed, output_size=(16, 16))  # (B, C, 16, 16)
                pooled_flat = pooled.flatten(2).transpose(1, 2)  # same as view+permute

                try:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        global_attended, _ = state_processor_dict['global_attention'](
                            pooled_flat, pooled_flat, pooled_flat
                        )
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print("⚠️ Warning: Global attention failed, using average pooling fallback.")
                        global_attended = pooled_flat.mean(dim=1, keepdim=True).expand(-1, pooled_flat.size(1), -1)
                    else:
                        raise e

                # Restore shape and upsample back to original H×W
                restored = global_attended.permute(0, 2, 1).view(B, C, 16, 16)
                processed = F.interpolate(restored, size=(H, W), mode='bilinear', align_corners=False)

                learning_rates.append(0.01)  # Standard rate for spatial
                
            else:
                processed = state_processor_dict['processor'](input_tensor)
                learning_rates.append(0.01)  # Standard rate
                
            processed_outputs.append(processed)
        
        # Weighted combination based on state probabilities (implementing Σᵢ State_weights × P_m^{state_i})
        combined_output = torch.zeros_like(processed_outputs[0])
        for state_idx, (output, alpha) in enumerate(zip(processed_outputs, learning_rates)):
            weight = new_state_probs[:, state_idx:state_idx+1, None, None]
            # Apply state-dependent learning rate α_m^{state_i}
            state_contribution = alpha * weight * output
            combined_output += state_contribution
        
        # Update memory while preserving spatial structure
        memory_features = self.memory_integration(combined_output)
        
        # Apply residual connection with current memory state
        integrated_memory = self.memory_projection(memory_features)
        updated_memory = 0.9 * memory_state + 0.1 * integrated_memory
        if self.memory_type == "spatial" and state_idx == 2:
            del pooled, pooled_flat, global_attended, restored
            torch.cuda.empty_cache() # Clear cache for large tensors
        
        return updated_memory, new_state_probs