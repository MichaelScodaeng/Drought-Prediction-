import torch
from torch.utils.data import DataLoader
from typing import Dict
from src.mesanet.mesanet import MESANetLayer
from src.mesanet.mesanet_loss import MESANetLoss
from src.mesanet.mesanet_dataset import WeatherBench2Dataset
from src.mesanet.mesanet_datamanager import WeatherBench2DataManager

class MESANetTrainer:
    """Training pipeline for MESA-Net"""
    
    def __init__(self,
                 model: MESANet,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 loss_fn: MESANetLoss,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 save_dir: str = "./mesa_net_checkpoints"):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        
        # Training metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.state_histories = []
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'prediction_loss': 0.0,
            'state_entropy_loss': 0.0,
            'transition_smooth_loss': 0.0,
            'cross_memory_loss': 0.0,
            'cross_layer_loss': 0.0,
            'total_loss': 0.0
        }
        
        num_batches = 0
        
        for batch_idx, (input_seq, target_seq, geo_features) in enumerate(self.train_loader):
            # Move data to device
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)
            geo_features = geo_features.to(self.device)
            
            # Forward pass
            predictions, states_history = self.model(
                input_seq, geo_features, forecast_steps=target_seq.size(1)
            )
            
            # Compute loss
            total_loss, loss_components = self.loss_fn(
                predictions, target_seq, states_history, 
                states_history['memory_states'][-1]
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in loss_components.items():
                epoch_losses[key] += value.item()
            
            num_batches += 1
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {total_loss.item():.4f}")
        
        # Average losses over epoch
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate model performance"""
        self.model.eval()
        val_losses = {
            'prediction_loss': 0.0,
            'state_entropy_loss': 0.0,
            'transition_smooth_loss': 0.0,
            'cross_memory_loss': 0.0,
            'cross_layer_loss': 0.0,
            'total_loss': 0.0
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (input_seq, target_seq, geo_features) in enumerate(self.val_loader):
                # Limit validation batches for faster validation
                if batch_idx >= 20:
                    break
                    
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                geo_features = geo_features.to(self.device)
                
                predictions, states_history = self.model(
                    input_seq, geo_features, forecast_steps=target_seq.size(1)
                )
                
                total_loss, loss_components = self.loss_fn(
                    predictions, target_seq, states_history,
                    states_history['memory_states'][-1]
                )
                
                for key, value in loss_components.items():
                    val_losses[key] += value.item()
                
                num_batches += 1
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches if num_batches > 0 else 1
        
        return val_losses
    
    def train(self, num_epochs: int):
        """Complete training loop"""
        import os
        os.makedirs(self.save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            
            # Training
            train_losses = self.train_epoch(epoch)
            self.train_losses.append(train_losses)
            
            print(f"Train Loss: {train_losses['total_loss']:.4f}")
            print(f"  Prediction: {train_losses['prediction_loss']:.4f}")
            print(f"  State Entropy: {train_losses['state_entropy_loss']:.4f}")
            print(f"  Transition Smooth: {train_losses['transition_smooth_loss']:.4f}")
            
            # Validation
            val_losses = self.validate()
            self.val_losses.append(val_losses)
            
            print(f"Val Loss: {val_losses['total_loss']:.4f}")
            
            # Save best model
            if val_losses['total_loss'] < best_val_loss:
                best_val_loss = val_losses['total_loss']
                self.save_checkpoint(epoch, is_best=True)
                print(f"New best model saved! Val Loss: {best_val_loss:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        import os
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        if is_best:
            path = os.path.join(self.save_dir, 'best_model.pth')
        else:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from epoch {epoch}")
        return epoch