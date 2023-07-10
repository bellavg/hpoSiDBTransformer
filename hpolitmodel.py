import pytorch_lightning as pl
import torch
from focal_loss.focal_loss import FocalLoss
from transformer import SiDBTransformer
from hyperparameters import *
import gc
import torch.nn as nn


def get_accuracy(outputs, targets):
    print(outputs, "o")
    targets = targets.reshape(-1)
    mask = targets >= 0
    outputs = outputs.permute(0, 2, 3, 1).reshape((-1, 2))
    masked_target = targets[mask]
    masked_output = outputs[mask.unsqueeze(-1).repeat(1, 2)].view(-1, 2)
    pred = masked_output.argmax(dim=1, keepdim=True)
    print(pred, pred.shape)
    has_zero_class = (pred == 0).any()
    #assert has_zero_class.item() is False, "no zero class"
    has_one_class = (pred == 1).any()
    #assert has_one_class.item() is False, "no zero class"
    accuracy = torch.mean((pred == masked_target).float())
    return accuracy, pred, masked_target


class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.transformer = SiDBTransformer(input_dim=INPUTCHANNELS, position_info=config["pi"],
                                           depth=config["depth"], embeddim=config["embedding_dim"],
                                           heads=config["head"],
                                           gridsize=GRIDSIZE, d_rate=config["dropout"])
        self.opname = "Adam"
        self.lr = config["lr"]
        self.wd = config["weight_decay"]
        self.lossfn = nn.NLLLoss(ignore_index=-1)

    def forward(self, x):
        return self.transformer(x)

    def training_step(self, batch, batch_idx):
        x, targets = batch
        targets = targets.to(x.device)
        outputs = self(x)
        loss = self.lossfn(outputs, targets)  # check sizes should be b, 2, 42, 42 and b, 42, 42
        self.log("train_loss", loss, logger=True, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        x.to(self.device)
        targets.to(self.device)
        outputs = self.transformer(x)
        accuracy, _, _ = get_accuracy(outputs, targets)
        self.log("val_acc", accuracy, sync_dist=True, logger=True, on_epoch=True, on_step=False)
        self.log("hp_metric", accuracy, on_step=False, on_epoch=True, sync_dist=True)
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        optimizer = getattr(
            torch.optim, self.opname
        )(self.transformer.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer

