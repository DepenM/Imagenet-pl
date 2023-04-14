import os
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax
from torchmetrics import Accuracy
from torchvision import models


def get_model(model_name):
    return models.__dict__[model_name.lower()](weights=None, num_classes=1000)


class LitModel(pl.LightningModule):

    def __init__(self,
                 arch: str = "resnet50",
                 optim: str = "sgd",
                 learning_rate: float = 1e-1,
                 weight_decay: float = 1e-4,
                 momentum: float = 0.9,
                 max_steps: int = 90*626, # 56340 
                 schedule: str = 'cos'):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model = get_model(arch)
        #move model to channels last
        self.model = self.model.to(memory_format=torch.channels_last)

        self.train_acc = Accuracy()
        self.pred_accs = Accuracy()

        self.learning_rate = learning_rate
        self.optim = optim
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.max_steps = max_steps

        self.schedule = schedule

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: Optional[int] = None):
        images, _ = batch
        return self.model(images)

    def process_batch(self, batch, stage="train"):
        images, labels = batch
        logits = self.forward(images)
        probs = softmax(logits, dim=1)
        loss = self.criterion(logits, labels)

        if stage == "train":
            self.train_acc(probs, labels)
        elif stage == "pred":
            self.pred_accs(probs, labels)
        else:
            raise ValueError("Invalid stage %s" % stage)

        return loss

    def training_step(self, batch, batch_idx: int):
        loss = self.process_batch(batch, "train")
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx: int):
        loss = self.process_batch(batch, "pred")
        self.log("pred_loss", loss, sync_dist=True)
        self.log("pred_acc", self.pred_accs, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        parameters = self.model.parameters()

        if self.optim == "sgd":
            optimizer = torch.optim.SGD(parameters,
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_decay,
                                        momentum=self.momentum)
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(parameters,
                                         lr=self.learning_rate,
                                         weight_decay=self.weight_decay)
        elif self.optim == "adamw":
            optimizer = torch.optim.AdamW(parameters,
                                          lr=self.learning_rate,
                                          weight_decay=self.weight_decay)
        elif self.optim == "lamb":
            from timm.optim import Lamb
            optimizer = Lamb(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError("Unknown optimizer %s" % self.optim)

        if self.schedule == 'step':
            lr_scheduler = lr_sched.StepLR(optimizer, step_size=self.max_steps // 3 + 1, gamma=0.1)
        elif self.schedule == 'cos':
            lr_scheduler = lr_sched.CosineAnnealingLR(optimizer, T_max=self.max_steps)
        else:
            raise NotImplementedError()

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "frequency": 1, # every epoch with batch size 512 * 4
                "interval": "step",
            },
        }