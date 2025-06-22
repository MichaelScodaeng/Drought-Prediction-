# MINIMAL FIXES for mesanet_trainer.py
# Just replace the problematic methods with these fixed versions
import traceback

import os
import torch
import time  # Add this import
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from typing import Dict, Optional
from torch.cuda.amp import autocast, GradScaler
from src.mesanet.mesanet import MESANet
from src.mesanet.mesanet_loss import MESANetLoss

class MESANetTrainer:
    def __init__(self,
                 model: MESANet,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 loss_fn: MESANetLoss,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 save_dir: str = "./mesa_net_checkpoints",
                 max_val_batches: int = 20,
                 print_every: int = 5,  # 🔧 FIX: More frequent updates
                 use_tensorboard: bool = False):  # 🔧 FIX: Disable by default for speed

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.max_val_batches = max_val_batches
        self.print_every = print_every
        os.makedirs(self.save_dir, exist_ok=True)

        self.train_losses = []
        self.val_losses = []

        self.writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs')) if use_tensorboard else None
        self.scaler = GradScaler()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """🔧 FIXED: Added error handling and better progress tracking"""
        self.model.train()
        losses = self._init_loss_dict()
        num_batches = 0
        
        # 🔧 FIX: Add timing for performance monitoring
        epoch_start = time.time()
        successful_batches = 0
        failed_batches = 0

        for i, batch_data in enumerate(self.train_loader):
            batch_start = time.time()
            
            try:
                # 🔧 FIX: Safe batch unpacking
                if len(batch_data) != 3:
                    #print(f"⚠️ Skipping batch {i}: unexpected format")
                    continue
                    
                x, y, geo = batch_data
                x, y, geo = x.to(self.device), y.to(self.device), geo.to(self.device)
                self.optimizer.zero_grad()
                #print(f"🔄 Processing batch {i+1}/{len(self.train_loader)}...")
                #print("x,y,geo: ", x.shape, y.shape, geo.shape)

                # 🔧 FIX: Proper mixed precision handling
                device_type = "cuda" if torch.cuda.is_available() else "cpu"
                with autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
                    pred, state_hist = self.model(x, geo, forecast_steps=y.size(1))

                # 🔧 FIX: Safe memory state access
                with autocast(enabled=False):
                    last_memory_states = {}
                    if (state_hist and 'memory_states' in state_hist and 
                        len(state_hist['memory_states']) > 0):
                        last_memory_states = state_hist['memory_states'][-1]
                    
                    loss, components = self.loss_fn(pred.float(), y.float(), state_hist, last_memory_states)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Track losses
                for k, v in components.items():
                    if k in losses:
                        losses[k] += v.item()
                
                num_batches += 1
                successful_batches += 1
                
                # 🔧 FIX: Better progress reporting with timing
                if i % self.print_every == 0:
                    batch_time = time.time() - batch_start
                    elapsed = time.time() - epoch_start
                    batches_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                    eta_minutes = (len(self.train_loader) - i - 1) / batches_per_sec / 60 if batches_per_sec > 0 else 0
                    
                    print(f"   Batch {i+1}/{len(self.train_loader)} | "
                          f"Loss: {loss.item():.4f} | "
                          f"Time: {batch_time:.2f}s | "
                          f"ETA: {eta_minutes:.1f}min")
                    
                    # Memory usage
                    if torch.cuda.is_available():
                        memory_gb = torch.cuda.memory_allocated() / 1e9
                        #print(f"   GPU Memory: {memory_gb:.1f}GB")

            except Exception as e:
                failed_batches += 1
                #print(f"❌ Error in batch {i}: {str(e)[:100]}...")
                traceback.print_exc()
                # Clean up GPU memory after error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Skip this batch and continue
                continue

        epoch_time = time.time() - epoch_start
        #print(f"✅ Epoch {epoch+1} completed: {successful_batches} successful, {failed_batches} failed batches")
        #print(f"   Total time: {epoch_time/60:.1f} minutes")

        return self._average_losses(losses, num_batches)

    def validate(self) -> Dict[str, float]:
        """🔧 FIXED: Added error handling for validation"""
        self.model.eval()
        losses = self._init_loss_dict()
        num_batches = 0
        
        #print(f"🔍 Starting validation ({self.max_val_batches} batches max)...")
        val_start = time.time()

        with torch.no_grad():
            for i, batch_data in enumerate(self.val_loader):
                if i >= self.max_val_batches:
                    break
                
                try:
                    if len(batch_data) != 3:
                        continue
                        
                    x, y, geo = batch_data
                    x, y, geo = x.to(self.device), y.to(self.device), geo.to(self.device)
                    
                    # 🔧 FIX: Proper mixed precision
                    device_type = "cuda" if torch.cuda.is_available() else "cpu"
                    with autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
                        pred, state_hist = self.model(x, geo, forecast_steps=y.size(1))

                    # 🔧 FIX: Safe memory state access
                    with autocast(enabled=False):
                        last_memory_states = {}
                        if (state_hist and 'memory_states' in state_hist and 
                            len(state_hist['memory_states']) > 0):
                            last_memory_states = state_hist['memory_states'][-1]
                        
                        loss, components = self.loss_fn(pred.float(), y.float(), state_hist, last_memory_states)
                    
                    for k, v in components.items():
                        if k in losses:
                            losses[k] += v.item()
                    
                    num_batches += 1
                    
                    if i % 5 == 0:
                        print(f"   Val batch {i+1}")

                except Exception as e:
                    #print(f"❌ Validation error in batch {i}: {str(e)[:50]}...")
                    continue

        val_time = time.time() - val_start
        #print(f"✅ Validation completed in {val_time:.1f}s ({num_batches} batches)")

        return self._average_losses(losses, num_batches)

    def train(self, num_epochs: int, early_stopping_patience: Optional[int] = 5):
        """🔧 ENHANCED: Better training loop with state analysis"""
        #print(f"🚀 Starting MESA-Net Training")
        #print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        #print(f"   Device: {self.device}")
        #print(f"   Epochs: {num_epochs}")
        
        best_val = float('inf')
        patience = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            #print(f"\n{'='*60}")
            #print(f"📅 EPOCH {epoch + 1}/{num_epochs}")
            #print(f"{'='*60}")
            
            try:
                # Training
                train_loss = self.train_epoch(epoch)
                
                # Validation
                val_loss = self.validate()
                
                # Store losses
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)

                # 🔧 ENHANCED: Better logging with state analysis
                self._log_epoch_enhanced(epoch, train_loss, val_loss)

                # Check for improvement
                if val_loss['total_loss'] < best_val:
                    best_val = val_loss['total_loss']
                    patience = 0
                    self.save_checkpoint(epoch, is_best=True)
                    #print(f"🎉 New best model saved! Val Loss: {best_val:.4f}")
                else:
                    patience += 1
                    if early_stopping_patience and patience >= early_stopping_patience:
                        #print(f"⏹️ Early stopping triggered after {patience} epochs without improvement")
                        break

                # Periodic checkpoint
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(epoch)

                # Memory cleanup
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated() / 1e9
                    #print(f"🖥️ GPU Peak Memory: {peak_memory:.2f} GB")
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

            except Exception as e:
                #print(f"❌ Error in epoch {epoch + 1}: {e}")
                #print("Continuing to next epoch...")
                continue

        total_time = time.time() - start_time
        #print(f"\n🏁 Training completed in {total_time/3600:.1f} hours")
        
        if self.writer:
            self.writer.close()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """🔧 ENHANCED: Better checkpoint saving with error handling"""
        try:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),  # Save scaler state
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
            }
            
            filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
            path = os.path.join(self.save_dir, filename)
            
            torch.save(checkpoint_data, path)
            print(f"💾 Checkpoint saved: {path}")
            
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")

    def _init_loss_dict(self) -> Dict[str, float]:
        return {
            'prediction_loss': 0.0,
            'state_entropy_loss': 0.0,
            'transition_smooth_loss': 0.0,
            'cross_memory_loss': 0.0,
            'cross_layer_loss': 0.0,
            'total_loss': 0.0
        }

    def _average_losses(self, losses: Dict[str, float], count: int) -> Dict[str, float]:
        return {k: v / max(count, 1) for k, v in losses.items()}

    def _log_epoch_enhanced(self, epoch: int, train_loss: Dict[str, float], val_loss: Dict[str, float]):
        """🔧 ENHANCED: Better epoch logging"""
        #print(f"\n📊 EPOCH {epoch+1} RESULTS:")
        #print("-" * 50)
        for k in train_loss:
            #print(f"{k:<22} | Train: {train_loss[k]:.6f} | Val: {val_loss[k]:.6f}")
            
            # Log to tensorboard if available
            if self.writer:
                self.writer.add_scalar(f"train/{k}", train_loss[k], epoch)
                self.writer.add_scalar(f"val/{k}", val_loss[k], epoch)

    # Keep your original _log_epoch method for compatibility
    def _log_epoch(self, epoch: int, train_loss: Dict[str, float], val_loss: Dict[str, float]):
        #print(f"Epoch {epoch+1} Results:")
        for k in train_loss:
            #print(f"  {k:<22} Train: {train_loss[k]:.6f} | Val: {val_loss[k]:.6f}")
            if self.writer:
                self.writer.add_scalar(f"train/{k}", train_loss[k], epoch)
                self.writer.add_scalar(f"val/{k}", val_loss[k], epoch)