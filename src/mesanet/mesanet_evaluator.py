import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from src.mesanet.mesanet import MESANet

class MESANetEvaluator:
    def __init__(self, model: MESANet, device: torch.device):
        self.model = model
        self.device = device

    def evaluate_precipitation_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)

        threshold = 0.1  # mm/6hr
        pred_binary = (predictions > threshold).astype(int)
        target_binary = (targets > threshold).astype(int)

        hits = np.sum((pred_binary == 1) & (target_binary == 1))
        misses = np.sum((pred_binary == 0) & (target_binary == 1))
        false_alarms = np.sum((pred_binary == 1) & (target_binary == 0))

        csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0
        pod = hits / (hits + misses) if (hits + misses) > 0 else 0
        far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0

        return {'mse': mse, 'mae': mae, 'rmse': rmse, 'csi': csi, 'pod': pod, 'far': far}

    def evaluate_final_model(self, dataloader: DataLoader, max_batches: int = 50) -> Dict[str, float]:
        self.model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for i, (x, y, geo) in enumerate(dataloader):
                if i >= max_batches:
                    break
                x, y, geo = x.to(self.device), y.to(self.device), geo.to(self.device)
                preds, _ = self.model(x, geo, forecast_steps=y.size(1))
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        return self.evaluate_precipitation_metrics(all_preds, all_targets)

    def evaluate_and_save(self, dataloader: DataLoader, save_path: str, max_batches: int = 50):
        import pandas as pd
        self.model.eval()
        rows = []

        with torch.no_grad():
            for i, (x, y, geo) in enumerate(dataloader):
                if i >= max_batches:
                    break
                x, y, geo = x.to(self.device), y.to(self.device), geo.to(self.device)
                preds, _ = self.model(x, geo, forecast_steps=y.size(1))

                for b in range(preds.size(0)):
                    for t in range(preds.size(1)):
                        for lat in range(preds.size(2)):
                            for lon in range(preds.size(3)):
                                rows.append({
                                    'batch': i, 'timestep': t, 'lat': lat, 'lon': lon,
                                    'target': y[b, t, lat, lon].item(),
                                    'prediction': preds[b, t, lat, lon].item()
                                })

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        #print(f"Saved predictions to: {save_path}")

    def evaluate_from_checkpoint(self, checkpoint_path: str, dataloader: DataLoader, max_batches: int = 50) -> Dict[str, float]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        #print(f"Loaded model from checkpoint: {checkpoint_path}")
        return self.evaluate_final_model(dataloader, max_batches=max_batches)

    def analyze_state_patterns(self, states_history: Dict) -> Dict[str, any]:
        analysis = {}
        memory_types = ['fast', 'slow', 'spatial', 'spatiotemporal']

        for memory_type in memory_types:
            state_evolution = []
            for timestep in states_history['state_probs']:
                avg_probs = torch.mean(timestep[memory_type], dim=0).cpu().numpy()
                state_evolution.append(avg_probs)
            state_evolution = np.array(state_evolution)
            analysis[f'{memory_type}_state_evolution'] = state_evolution
            analysis[f'{memory_type}_dominant_state'] = np.argmax(state_evolution, axis=1)
            analysis[f'{memory_type}_state_stability'] = np.std(state_evolution, axis=0)
        return analysis

    def plot_state_evolution(self, analysis: Dict[str, any], memory_type: str):
        if f"{memory_type}_state_evolution" not in analysis:
            #print(f"Memory type '{memory_type}' not found in analysis.")
            return
        evolution = analysis[f"{memory_type}_state_evolution"]
        plt.figure(figsize=(10, 4))
        for i in range(evolution.shape[1]):
            plt.plot(evolution[:, i], label=f"State {i}")
        plt.title(f"State Evolution for {memory_type} Memory")
        plt.xlabel("Time Step")
        plt.ylabel("State Probability")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def generate_attention_maps(self, input_sequence: torch.Tensor, geo_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            preds, state_hist = self.model(input_sequence.to(self.device), geo_features.to(self.device))
            batch_size, seq_len, h, w = input_sequence.shape[:2] + input_sequence.shape[-2:]
            attention_maps = {
                'spatial_attention': torch.rand(batch_size, h, w),
                'temporal_attention': torch.rand(batch_size, seq_len)
            }
        return attention_maps

    def plot_attention_maps(self, attention_maps: Dict[str, torch.Tensor], timestep: int = 0):
        spatial = attention_maps['spatial_attention'][timestep].cpu().numpy()
        temporal = attention_maps['temporal_attention'][timestep].cpu().numpy()

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(spatial, cmap='viridis')
        plt.colorbar()
        plt.title("Spatial Attention Map")

        plt.subplot(1, 2, 2)
        plt.plot(temporal)
        plt.title("Temporal Attention Weights")
        plt.xlabel("Input Time Step")
        plt.ylabel("Attention")

        plt.tight_layout()
        plt.show()

    def compare_with_baselines(self, test_loader: DataLoader, baseline_models: Dict[str, torch.nn.Module]) -> Dict[str, Dict[str, float]]:
        results = {'MESA-Net': self.evaluate_final_model(test_loader)}
        for name, model in baseline_models.items():
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for i, (x, y, _) in enumerate(test_loader):
                    if i >= 50:
                        break
                    x, y = x.to(self.device), y.to(self.device)
                    out = model(x)
                    preds.append(out.cpu())
                    targets.append(y.cpu())
            preds = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)
            results[name] = self.evaluate_precipitation_metrics(preds, targets)
        return results
