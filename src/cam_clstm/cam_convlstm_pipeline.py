
import os
import torch
import optuna
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from src.cam_clstm.causal_clsm_model import MyConvLSTMModel
from src.cam_clstm.gridded_multitask_seq2seq_dataset import GriddedMultitaskSeq2SeqDataset
from src.cam_clstm.causal_multitask_lightning_module import CausalMultitaskLightningModule
from src.data_utils import load_config, load_and_prepare_data, split_data_chronologically
from src.preprocess_utils import scale_data, inverse_transform_predictions
from src.feature_utils import engineer_features
from torch.utils.data import DataLoader


class CAMConvLSTMPipeline:
    def __init__(self, config_path):
        self.cfg = load_config(config_path)
        seed_everything(self.cfg['seed'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.lightning_module = None
        self.scaler_x, self.scaler_y = None, None
        self.run_output_dir = self.cfg['paths']['run_output_dir']
        os.makedirs(self.run_output_dir, exist_ok=True)

    def prepare_data(self):
        df = load_and_prepare_data(self.cfg)
        df = engineer_features(df, self.cfg)
        df_train, df_val, df_test = split_data_chronologically(df, self.cfg)

        (X_train, y_train_dict), self.scaler_x, self.scaler_y = scale_data(df_train, return_dict=True)
        (X_val, y_val_dict), _, _ = scale_data(df_val, self.scaler_x, self.scaler_y, return_dict=True)
        (X_test, y_test_dict), _, _ = scale_data(df_test, self.scaler_x, self.scaler_y, return_dict=True)

        dataset_train = GriddedMultitaskSeq2SeqDataset(X_train, y_train_dict)
        dataset_val = GriddedMultitaskSeq2SeqDataset(X_val, y_val_dict)
        dataset_test = GriddedMultitaskSeq2SeqDataset(X_test, y_test_dict)

        self.train_loader = DataLoader(dataset_train, batch_size=self.cfg['training']['batch_size'], shuffle=True)
        self.val_loader = DataLoader(dataset_val, batch_size=self.cfg['training']['batch_size'])
        self.test_loader = DataLoader(dataset_test, batch_size=self.cfg['training']['batch_size'])

    def build_model(self, trial=None):
        hidden_channels = trial.suggest_int("hidden_channels", 16, 64)
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        use_pos_enc = trial.suggest_categorical("use_pos_enc", [True, False])
        use_spatial_attn = trial.suggest_categorical("use_spatial_attn", [True, False])

        self.model = MyConvLSTMModel(
            input_channels=self.cfg['model']['input_channels'],
            height=self.cfg['grid']['height'],
            width=self.cfg['grid']['width'],
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            pos_channels=self.cfg['model'].get("pos_channels", 4),
            use_pos_enc=use_pos_enc,
            use_spatial_attn=use_spatial_attn,
            use_temporal_only=False,
            causal_masks=None
        )

        # Inject tuned learning rate into config for LightningModule
        self.cfg['training']['learning_rate'] = learning_rate
        self.lightning_module = CausalMultitaskLightningModule(self.model, self.cfg)
        

    def train(self):
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            ModelCheckpoint(dirpath=self.run_output_dir, monitor="val_loss", save_top_k=1, mode="min"),
            LearningRateMonitor(logging_interval="epoch")
        ]

        trainer = Trainer(
            max_epochs=self.cfg['training']['epochs'],
            callbacks=callbacks,
            default_root_dir=self.run_output_dir,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            precision="16-mixed" if self.cfg['training'].get("mixed_precision") else 32,
            log_every_n_steps=10
        )

        trainer.fit(self.lightning_module, self.train_loader, self.val_loader)

    def evaluate(self):
        trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu")
        results = trainer.test(self.lightning_module, dataloaders=self.test_loader)
        return results

    def tune_and_train(self, n_trials=20):
        def objective(trial):
            self.prepare_data()
            self.build_model(trial)
            self.train()
            val_loss = self.lightning_module.trainer.callback_metrics["val_loss"].item()
            return val_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        print("Best trial:", study.best_trial.params)
        best_config_path = os.path.join(self.run_output_dir, "best_config.yaml")
        with open(best_config_path, "w") as f:
            import yaml
            yaml.safe_dump(study.best_trial.params, f)

    def predict_on_full_data(self):
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                x = batch["x"].to(self.device)
                preds = self.model(x)
                for task in preds:
                    pred_np = preds[task].cpu().numpy()  # [B, 1, H, W]
                    for p in pred_np:
                        predictions.append({
                            "task": task,
                            "pred": p.squeeze()
                        })
        return predictions
