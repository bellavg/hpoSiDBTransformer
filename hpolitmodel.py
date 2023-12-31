import pytorch_lightning as pl
import torch
from focal_loss.focal_loss import FocalLoss
from transformer import SiDBTransformer
from hyperparameters import *
import gc
import torch.nn as nn


def get_accuracy(outputs, targets):
    mask = targets >= 0
    masked_target = targets[mask]
    masked_output = outputs[mask.unsqueeze(-1).repeat(1, 2)].view(-1, 2)
    pred = torch.argmax(masked_output, dim=1)
    accuracy = torch.mean((pred == masked_target).float())
    return accuracy, pred, masked_target


class LitModel(pl.LightningModule):
    def __init__(self,  config):
        super().__init__()
        self.transformer = SiDBTransformer( position_info=config["pi"], input_dim=INPUTCHANNELS,
                                           depth=config["depth"], embeddim=config["embedding_dim"],
                                           heads=config["head"],
                                           gridsize=GRIDSIZE, d_rate=config["dropout"])
        self.opname = "Adam"
        self.lr = config["lr"]
        self.wd = config["weight_decay"]
        self.lossfn = FocalLoss(gamma=2.0, ignore_index=-1)

    def training_step(self, batch, batch_idx):
        x, targets = batch
        targets = targets.reshape(-1).to(x.device)
        outputs = self.transformer(x)
        loss = self.lossfn(outputs, targets)  # check sizes should be b, 2, 42, 42 and b, 42, 42
        self.log("train_loss", loss, logger=True, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        x.to(self.device)
        targets = targets.reshape(-1).to(x.device)
        outputs = self.transformer(x)
        accuracy, _, _ = get_accuracy(outputs, targets)
        self.log("val_acc", accuracy, sync_dist=True, logger=True, on_epoch=True, on_step=False)


    def configure_optimizers(self):
        optimizer = getattr(
            torch.optim, self.opname
        )(self.transformer.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer

    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        x, targets = batch
        outputs = self.transformer(x)
        accuracy, pred, masked_target = get_accuracy(outputs, targets)
        self.log("test accuracy", accuracy, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        self.testpred.append(torch.Tensor.cpu(pred))
        self.testtarget.append(torch.Tensor.cpu(masked_target))
        return accuracy

