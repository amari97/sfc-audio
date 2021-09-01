import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Tuple, Dict

from torchmetrics import Accuracy, MetricCollection, Precision, Recall, ConfusionMatrix, F1

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torch.distributions as dist

from preprocessing.helpers import boolean_string
import seaborn as sns
import pandas as pd
import os
plt.style.use('ggplot')


class BaseModel(pl.LightningModule):
    """Basic model that set up the training loop and the optimization procedure.
    _forward_impl(self, x: torch.Tensor) -> torch.Tensor must be implemented in the child classes
    """

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Add arguments 
            -   lr : learning rate (default 0.005)
            -   weight_decay: if the weight decay should be used (default 0.0)
            -   sgd: if the stochastic gradient descent algorithm should be used (default true), otherwise use AdamW
            -   momentum: the momentum parameters (default 0.0)
        to the parser."""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # add here parameters of the model
        parser.add_argument('--lr', default=0.005, type=float)
        parser.add_argument('--weight_decay', default=0.0, type=float)
        parser.add_argument('--sgd', default=True, type=boolean_string)
        parser.add_argument('--momentum', default=0, type=float)
        return parser

    def __init__(self, lr: float = 0.005, weight_decay: float = 0, num_classes: int = 35, class_names: List = None, sgd: bool = True, mixup: bool = False, alpha: float = 0.2, smoothing: float = 0.0, momentum: float = 0.0, **params) -> None:
        """
        Args:
            lr (float): the learning rate
            weight_decay (float, positive): if the weight decay should be used (default 0.0)
            num_classes (int): the number of classes to use (default 35)
            class_names (list): the name of the classes (the length must be the number of classes)
            sgd (bool): if the stochastic gradient descent algorithm should be used (default true), otherwise use AdamW
            mixup (bool): if mixup should be used (default false)
            alpha (float): the mixup hyperparameter that defines the sampling distribution of lambda (default 0.2)
            momentum (float): the momentum parameters (default 0.0)
            smoothing (float): label smoothing parameters (default 0)
        """
        super().__init__()
        if class_names is not None:
            assert num_classes == len(class_names)
        metrics = MetricCollection({"accuracy": Accuracy(dist_sync_on_step=True, compute_on_step=False),
                                    "avg_precision": Precision(num_classes=num_classes, average='weighted', dist_sync_on_step=True, compute_on_step=False),
                                    "avg_recall": Recall(num_classes=num_classes, average='weighted', dist_sync_on_step=True, compute_on_step=False),
                                    "confusion_matrix": ConfusionMatrix(num_classes=num_classes, normalize="true", dist_sync_on_step=True, compute_on_step=False),
                                    "f1": F1(num_classes=num_classes, average="none", dist_sync_on_step=True, compute_on_step=False),
                                    })
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        self.beta = dist.beta.Beta(alpha, alpha)
        self.criterion = nn.CrossEntropyLoss()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # output size k x nb_classes
        output = self._forward_impl(x)
        return output

    def training_step(self, batch: Tuple, batch_idx) -> Dict:
        """Training loop with mixup if mixup is set to true"""
        x, y = batch
        # reshape k x 1 x x.size(2) x x.size(3)
        # set the number of channels to 1
        nb_repeat = x.size(1)
        x = x.reshape(-1, 1, x.size(2), x.size(3))
        y = y.repeat_interleave(nb_repeat)

        if self.hparams.mixup:
            # mixup procedure
            if x.size(0) < 2:
                raise ValueError("Batch size must be larger or equal to 2")
            if x.size(0) % 2 == 1:
                # remove last example
                x, y = x[:-1], y[:-1]
            # split data into two parts
            half = int(x.size(0)/2)
            x1, y1 = x[:half], y[:half]
            x2, y2 = x[half:], y[half:]
            l = self.beta.sample()
            # take linear combination of both parts
            x_ = l*x1+(1-l)*x2
            # negative log-likelihood (assume we have already used softmax)
            logits = self(x_)
            # nll linear in the second argument (same as l*y1+(1-l)*y2, if y1 and y2 are one-hot encoded)
            loss = l*self.criterion(logits, y1)+(1-l) * \
                self.criterion(logits, y2)

        else:
            # negative log-likelihood (assume we have already used softmax)
            logits = self(x)
            loss = self.criterion(logits, y)

        return {"loss": loss, "logits": logits, "target": y}

    def training_step_end(self, outs: Dict) -> torch.Tensor:
        # compute train metrics only if mixup is false
        if not self.hparams.mixup:
            self.train_metrics(torch.exp(outs["logits"]),  outs["target"])
        return outs["loss"].mean()

    def training_epoch_end(self, outs: Dict) -> torch.Tensor:
        # compute train metrics only if mixup is false
        if not self.hparams.mixup:
            self.compute_metrics(outs, "train")

    def compute_metrics(self, outs: Dict, case: str) -> None:
        # compute the metrics
        if case == "train":
            metrics = self.train_metrics.compute()
            self.train_metrics.reset()
        elif case == "val":
            metrics = self.valid_metrics.compute()
            self.valid_metrics.reset()
        elif case == "test":
            metrics = self.test_metrics.compute()
            self.test_metrics.reset()
        else:
            raise ValueError("Unknown case")

        # log the metrics
        for key, val in metrics.items():
            if key.endswith("accuracy"):
                self.log(key, val)
            if val.numel() == 1:
                self.log("metrics/"+key, val)
            elif key.endswith("confusion_matrix"):
                if key.startswith("test"):
                    # log confusion matrix only in the test case
                    fig = plt.figure(figsize=(18, 18))
                    df_cm = pd.DataFrame(val.cpu().numpy(
                    ), index=self.hparams.class_names, columns=self.hparams.class_names)
                    heatmap = sns.heatmap(
                        df_cm, annot=True, linewidths=.5, square=True, cbar=False)
                    heatmap.set_xticklabels(
                        heatmap.get_xticklabels(), rotation=90)
                    plt.xlabel("Predicted")
                    plt.ylabel("Target")
                    # check that the folder figure exists
                    os.makedirs("figure", exist_ok=True)
                    plt.savefig(os.path.join("figure", "confusion_matrix.png"))
            else:  # avg_recall/ avg_precision
                scalars = dict()
                for i in range(self.hparams.num_classes):
                    # name of the class if class_names is not none, or the index between 0 and num_classes
                    txt = self.hparams.class_names[i] if self.hparams.class_names is not None else str(
                        i)
                    scalars[txt] = val[i]
                    # register separately each metric
                    if case == "test":
                        self.log("metrics/{}_{}".format(key, txt), val[i])
                if case != "test":
                    self.logger.experiment.add_scalars(
                        "metrics/"+key, scalars, global_step=self.global_step)

    def validation_step(self, batch: Tuple, batch_idx) -> Dict:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return {"loss": loss, "logits": logits, "target": y}

    def validation_step_end(self, outs: Dict) -> torch.Tensor:
        loss = outs["loss"].mean()
        self.log('val_loss', loss, prog_bar=True)
        self.valid_metrics(torch.exp(outs["logits"]),  outs["target"])
        return loss

    def validation_epoch_end(self, outs: Dict) -> None:
        self.compute_metrics(outs, "val")

    def test_step(self, batch: Tuple, batch_idx) -> Dict:
        # same as validation step
        return self.validation_step(batch, batch_idx)

    def test_step_end(self, outs: Dict) -> torch.Tensor:
        loss = outs["loss"].mean()
        self.log("test_loss", loss)
        self.test_metrics(torch.exp(outs["logits"]),  outs["target"])
        return loss

    def test_epoch_end(self, outs: Dict) -> None:
        self.compute_metrics(outs, "test")

    def configure_optimizers(self) -> Dict:
        if self.hparams.sgd:
            optimizer = torch.optim.SGD(self.parameters(
            ), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum)
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30,T_mult=1)
        }

