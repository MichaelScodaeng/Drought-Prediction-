# optuna_tuning.py
import optuna
from train import train_adm_model  # Import the training function
from evaluate import evaluate_model  # Import the evaluation function
from data_utils import load_data  # Function to load your datasets

def objective(trial):
    """
    Objective function to be used by Optuna for hyperparameter optimization.
    """
    # Hyperparameters to tune
    hidden_channels = [trial.suggest_int('hidden_channels_1', 16, 128),
                       trial.suggest_int('hidden_channels_2', 16, 128)]
    n_layers = trial.suggest_int('n_layers', 2, 4)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    
    # Training configuration
    config = {
        'model': {
            'input_channels': 10,  # Adjust based on your data
            'hidden_channels': hidden_channels,
            'n_layers': n_layers,
            'output_targets': ["PET", "PRE"]
        },
        'training': {
            'batch_size': 16,
            'epochs': 50,
            'learning_rate': learning_rate,
            'use_dropout': dropout_rate > 0.0,  # Apply dropout if rate > 0
            'shuffle': True,
            'early_stopping_patience': 10
        },
        'output': {
            'save_model_path': 'best_model.pth'
        }
    }

    # Load the dataset
    train_loader, val_loader, land_mask = load_data(config)  # Define the data loading function

    # Train the model
    model = train_adm_model(train_loader, config, val_loader=val_loader, land_mask=land_mask)
    
    # Evaluate the model
    metrics = evaluate_model(model, val_loader, land_mask, output_targets=config['model']['output_targets'])
    
    # Return RMSE as the optimization target (minimize RMSE)
    return metrics['PET']['RMSE']  # Choose the appropriate target variable for tuning

def run_optuna_tuning():
    # Set up the Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)  # Perform 50 trials, adjust as needed

    # After tuning, print the best hyperparameters
    print("Best hyperparameters found by Optuna:")
    print(study.best_trial.params)
