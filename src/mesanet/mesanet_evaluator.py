import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple
from mesanet.mesanet import MESANet,MESANetLoss
from mesanet.mesanet_dataset import WeatherBench2Dataset
from mesanet.mesanet_trainer import MESANetTrainer
from mesanet.mesanet_datamanager import WeatherBench2DataManager
from dataclasses import dataclass
class MESANetEvaluator:
    """Evaluation and interpretation tools for MESA-Net"""
    
    def __init__(self, model: MESANet, device: torch.device):
        self.model = model
        self.device = device
        
    def evaluate_precipitation_metrics(self, 
                                     predictions: torch.Tensor,
                                     targets: torch.Tensor) -> Dict[str, float]:
        """Compute precipitation-specific evaluation metrics"""
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        
        # Basic metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        
        # Precipitation-specific metrics
        # Critical Success Index (CSI) for precipitation detection
        threshold = 0.1  # mm/6hr precipitation threshold
        pred_binary = (predictions > threshold).astype(int)
        target_binary = (targets > threshold).astype(int)
        
        hits = np.sum((pred_binary == 1) & (target_binary == 1))
        misses = np.sum((pred_binary == 0) & (target_binary == 1))
        false_alarms = np.sum((pred_binary == 1) & (target_binary == 0))
        
        csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0
        
        # Probability of Detection (POD)
        pod = hits / (hits + misses) if (hits + misses) > 0 else 0
        
        # False Alarm Rate (FAR)
        far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'csi': csi,
            'pod': pod,
            'far': far
        }
    
    def analyze_state_patterns(self, states_history: Dict) -> Dict[str, any]:
        """Analyze learned state patterns for interpretability"""
        analysis = {}
        
        # Extract state probabilities over time
        memory_types = ['fast', 'slow', 'spatial', 'spatiotemporal']
        
        for memory_type in memory_types:
            state_evolution = []
            for timestep in states_history['state_probs']:
                state_probs = timestep[memory_type]
                # Average over batch dimension
                avg_probs = torch.mean(state_probs, dim=0).cpu().numpy()
                state_evolution.append(avg_probs)
            
            state_evolution = np.array(state_evolution)  # Shape: (time, num_states)
            
            analysis[f'{memory_type}_state_evolution'] = state_evolution
            analysis[f'{memory_type}_dominant_state'] = np.argmax(state_evolution, axis=1)
            analysis[f'{memory_type}_state_stability'] = np.std(state_evolution, axis=0)
        
        return analysis
    
    def generate_attention_maps(self, 
                               input_sequence: torch.Tensor,
                               geo_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate attention maps for visualization"""
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass with hooks to capture intermediate attention
            predictions, states_history = self.model(input_sequence, geo_features)
            
            # Extract attention patterns from the last layer
            attention_maps = {}
            
            # This would require modifying the model to return attention weights
            # For now, return placeholder
            batch_size, seq_len, height, width = input_sequence.shape[:2] + input_sequence.shape[-2:]
            
            attention_maps['spatial_attention'] = torch.randn(batch_size, height, width)
            attention_maps['temporal_attention'] = torch.randn(batch_size, seq_len)
            
        return attention_maps
    
    def compare_with_baselines(self, 
                              test_loader: DataLoader,
                              baseline_models: Dict[str, nn.Module]) -> Dict[str, Dict[str, float]]:
        """Compare MESA-Net with baseline models"""
        results = {}
        
        # Evaluate MESA-Net
        mesa_metrics = self._evaluate_model(self.model, test_loader)
        results['MESA-Net'] = mesa_metrics
        
        # Evaluate baseline models
        for model_name, model in baseline_models.items():
            baseline_metrics = self._evaluate_model(model, test_loader)
            results[model_name] = baseline_metrics
        
        return results
    
    def _evaluate_model(self, model: nn.Module, data_loader: DataLoader) -> Dict[str, float]:
        """Helper function to evaluate any model"""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (input_seq, target_seq, geo_features) in enumerate(data_loader):
                if batch_idx >= 50:  # Limit evaluation for speed
                    break
                    
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                geo_features = geo_features.to(self.device)
                
                if hasattr(model, 'forward') and 'geo_features' in model.forward.__code__.co_varnames:
                    # MESA-Net style model
                    predictions, _ = model(input_seq, geo_features)
                else:
                    # Standard model
                    predictions = model(input_seq)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(target_seq.cpu())
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        return self.evaluate_precipitation_metrics(all_predictions, all_targets)
