
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

class CausalMultitaskLightningModule(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.learning_rate = cfg['training']['learning_rate']
        self.loss_fn = torch.nn.MSELoss()
        self.target_tasks = ['pre', 'pet', 'spei']

    def forward(self, x, temporal_only_input=None):
        return self.model(x, temporal_only_input=temporal_only_input)

    def training_step(self, batch, batch_idx):
        x, y_dict = batch["x"], batch["y"]
        temporal_only = batch.get("temporal_only", None)
        preds = self(x, temporal_only_input=temporal_only)

        loss, loss_dict = 0.0, {}
        for task in self.target_tasks:
            task_loss = self.loss_fn(preds[task], y_dict[task][:, -1])  # last step
            loss += task_loss
            loss_dict[f"{task}_loss"] = task_loss
        self.log_dict(loss_dict, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_dict = batch["x"], batch["y"]
        temporal_only = batch.get("temporal_only", None)
        preds = self(x, temporal_only_input=temporal_only)

        loss, loss_dict = 0.0, {}
        for task in self.target_tasks:
            task_loss = self.loss_fn(preds[task], y_dict[task][:, -1])
            loss += task_loss
            loss_dict[f"val_{task}_loss"] = task_loss
        self.log_dict(loss_dict, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y_dict = batch["x"], batch["y"]
        temporal_only = batch.get("temporal_only", None)
        preds = self(x, temporal_only_input=temporal_only)

        loss, loss_dict = 0.0, {}
        for task in self.target_tasks:
            task_loss = self.loss_fn(preds[task], y_dict[task][:, -1])
            loss += task_loss
            loss_dict[f"test_{task}_loss"] = task_loss
        self.log_dict(loss_dict, prog_bar=False)
        self.log("test_loss", loss)
        return {"preds": preds, "target": y_dict}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
