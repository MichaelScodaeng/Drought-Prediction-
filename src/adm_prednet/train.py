# train.py
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.adm_prednet.stacked_model import ADMStackedModel
from src.adm_prednet.masked_loss import masked_mse
from src.adm_prednet.pe_utils import add_temporal_pe, add_spatial_pe
import numpy as np
import pandas as pd
def train_adm_model(dataset, config, val_loader=None, land_mask=None, log_path=None):
    """
    Trains the ADMStackedModel with optional validation monitoring and early stopping.
    """
    # Set up model with configuration
    model = ADMStackedModel(
        input_channels=config['model']['input_channels'],
        hidden_channels=config['model']['hidden_channels'],
        n_layers=config['model']['n_layers'],
        output_targets=config['model']['output_targets'],
    ).cuda()
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Set up learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    if log_path != None:
        epochs = config['training']['epochs']
    else:
        epochs = 100
    best_val_loss = float('inf')
    patience = config['training'].get('early_stopping_patience', 5)
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for x_seq, y_true in dataset:
            x_seq, y_true = x_seq.cuda(), y_true.cuda()

            # Apply temporal and spatial positional encoding
            x_seq = add_temporal_pe(x_seq)
            x_seq = add_spatial_pe(x_seq)
            
            # Forward pass to get model predictions
            y_pred = model(x_seq)  # [B, C, H, W]
            
            # Compute the total loss across all targets
            output_targets = config['model']['output_targets']
            loss_weights = config['training'].get('loss_weights', {t: 1.0 for t in output_targets})

            # Weighted multitask loss
            loss_total = 0.0
            for i, target in enumerate(output_targets):
                pred = y_pred[:, i]
                true = y_true[:, i]
                loss = masked_mse(pred, true, land_mask)
                loss_total += loss_weights[target] * loss

            # Correct: use loss_total directly
            avg_train_loss = loss_total


            # Backpropagation and optimization
            optimizer.zero_grad()
            avg_train_loss.backward()
            optimizer.step()
            
            total_train_loss += avg_train_loss.item()
        
        avg_train_loss = total_train_loss / len(dataset)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
        
        # --- Validation ---
        model.eval()
        total_val_loss = 0
        if val_loader:
            with torch.no_grad():
                for x_seq, y_true in val_loader:
                    x_seq, y_true = x_seq.cuda(), y_true.cuda()
                    
                    # Apply temporal and spatial positional encodings for validation
                    x_seq = add_temporal_pe(x_seq)
                    x_seq = add_spatial_pe(x_seq)
                    
                    # Forward pass to get predictions
                    y_pred = model(x_seq)
                    
                    # Compute validation loss for all targets
                    output_targets = config['model']['output_targets']
                    loss_weights = config['training'].get('loss_weights', {t: 1.0 for t in output_targets})

                    val_loss_total = 0.0
                    for i, target in enumerate(output_targets):
                        pred = y_pred[:, i]
                        true = y_true[:, i]
                        loss = masked_mse(pred, true, land_mask)
                        val_loss_total += loss_weights[target] * loss
                    # Correct: use val_loss_total directly
                    total_val_loss += val_loss_total.item() 

            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")
            
            # Learning rate scheduler step
            scheduler.step(avg_val_loss)
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), config['output']['save_model_path'])
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        train_losses.append(avg_train_loss)
        if val_loader:
            val_losses.append(avg_val_loss)
    if log_path:

        log_df = pd.DataFrame({
            "epoch": list(range(1, len(train_losses)+1)),
            "train_loss": train_losses,
            "val_loss": val_losses if val_losses else [None]*len(train_losses)
        })
        
        log_df.to_csv(log_path, index=False)
        print(f"Training log saved to {log_path}")


    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model
