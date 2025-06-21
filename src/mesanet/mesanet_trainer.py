import os
import torch
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
                 print_every: int = 50,
                 use_tensorboard: bool = True):

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
        self.model.train()
        losses = self._init_loss_dict()
        num_batches = 0

        for i, (x, y, geo) in enumerate(self.train_loader):
            x, y, geo = x.to(self.device), y.to(self.device), geo.to(self.device)
            self.optimizer.zero_grad()

            with autocast(enabled=True, dtype=torch.float16):
                pred, state_hist = self.model(x, geo, forecast_steps=y.size(1))

            # Run loss in float32 for numerical stability
            with autocast(enabled=False):
                loss, components = self.loss_fn(pred.float(), y.float(), state_hist, state_hist['memory_states'][-1])

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k, v in components.items():
                losses[k] += v.item()
            losses['total_loss'] += loss.item()
            num_batches += 1

            if i % self.print_every == 0:
                print(f"Epoch {epoch+1} Batch {i}/{len(self.train_loader)} Loss: {loss.item():.4f}")

        return self._average_losses(losses, num_batches)

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        losses = self._init_loss_dict()
        num_batches = 0

        with torch.no_grad():
            for i, (x, y, geo) in enumerate(self.val_loader):
                if i >= self.max_val_batches:
                    break
                x, y, geo = x.to(self.device), y.to(self.device), geo.to(self.device)
                with autocast(enabled=True, dtype=torch.float16):
                    pred, state_hist = self.model(x, geo, forecast_steps=y.size(1))

                # Run loss in float32 for numerical stability
                with autocast(enabled=False):
                    loss, components = self.loss_fn(pred.float(), y.float(), state_hist, state_hist['memory_states'][-1])
                for k, v in components.items():
                    losses[k] += v.item()
                losses['total_loss'] += loss.item()
                num_batches += 1

        return self._average_losses(losses, num_batches)

    def train(self, num_epochs: int, early_stopping_patience: Optional[int] = 5):
        best_val = float('inf')
        patience = 0

        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self._log_epoch(epoch, train_loss, val_loss)

            if val_loss['total_loss'] < best_val:
                best_val = val_loss['total_loss']
                patience = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"New best model saved! Val Loss: {best_val:.4f}")
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    print("⏹️ Early stopping triggered.")
                    break

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)

            if torch.cuda.is_available():
                print(f"GPU Peak Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
                torch.cuda.reset_peak_memory_stats()

        if self.writer:
            self.writer.close()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        path = os.path.join(self.save_dir, 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        print(f"Checkpoint saved: {path}")

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

    def _log_epoch(self, epoch: int, train_loss: Dict[str, float], val_loss: Dict[str, float]):
        print(f"Epoch {epoch+1} Results:")
        for k in train_loss:
            print(f"  {k:<22} Train: {train_loss[k]:.6f} | Val: {val_loss[k]:.6f}")
            if self.writer:
                self.writer.add_scalar(f"train/{k}", train_loss[k], epoch)
                self.writer.add_scalar(f"val/{k}", val_loss[k], epoch)
