from dataclasses import dataclass
from typing import Any

import torch
import torchmetrics
from efficientnet_pytorch import EfficientNet
from pytorch_lightning import LightningModule
from sport_iseng.base_config import BaseConfig  # type: ignore
from torch.nn import functional as F


@dataclass
class LitImageArgs(BaseConfig):  # type: ignore
    """Config for LitImageClassifier

    Parameters
    ----------
    num_labels : int
        Number of label
    model_name: str
        Model name cv
    pretrained: bool
        whether the model is pretrained or not
    learning_rate: float
        Learning rate of the model
    """

    num_labels: int
    model_name: str = "effnetb0"
    is_pretrained: bool = False
    learning_rate: float = 1e-3


class LitImageClassifier(LightningModule):
    """
    Image classifier with pytorch lightning
    """

    def __init__(self, lit_args: LitImageArgs) -> None:
        """
        ~

        Parameters
        ----------
        lit_args : LitImageArgs
            Lit argument
        """
        super().__init__()
        self.save_hyperparameters()
        self.lit_args = lit_args
        self._learning_rate = lit_args.learning_rate
        self._instantiate_model()
        self._instantiate_metrics()
        self.output_layer = torch.nn.Linear(
            self.feature_shape, self.lit_args.num_labels
        )

    def _instantiate_metrics(self) -> None:
        """
        F1 and acc instantiate
        """
        self.acc_score = torchmetrics.Accuracy(num_classes=self.lit_args.num_labels)
        self.f1_score = torchmetrics.F1Score(num_classes=self.lit_args.num_labels)

    def _instantiate_model(self) -> None:
        """
        Model preparation to instantiate it
        """
        if self.lit_args.model_name == "effnetb0":
            self.feature_shape = 1000
            if self.lit_args.is_pretrained:
                self.backbone = EfficientNet.from_name("efficientnet-b0")
            else:
                self.backbone = EfficientNet.from_pretrained("efficientnet-b0")
        else:
            raise NotImplementedError("other model_name are not implemented")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        We use ADAM here
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x: torch.Tensor) -> Any:  # type: ignore  # pylint: disable=all
        """
        Return logits
        """
        forward_bb = self.backbone(x)
        out_logit = self.output_layer(forward_bb)
        return out_logit

    def training_step(self, batch, batch_idx):  # type: ignore
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.acc_score(y_hat, y)
        self.f1_score(y_hat, y)
        self.log("train_acc", self.acc_score, on_step=True, on_epoch=False)
        self.log("train_f1", self.f1_score, on_step=True, on_epoch=False)
        return loss

    def training_epoch_end(self, outputs):  # type: ignore
        self.log("train_epoch_acc", self.acc_score)
        self.log("train_epoch_f1", self.f1_score)

    def validation_step(self, batch, batch_idx):  # type: ignore
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.acc_score(y_hat, y)
        self.f1_score(y_hat, y)
        self.log("val_acc", self.acc_score, on_step=True, on_epoch=True)
        self.log("val_f1", self.f1_score, on_step=True, on_epoch=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):  # type: ignore
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.acc_score(y_hat, y)
        self.f1_score(y_hat, y)
        self.log("acc", self.acc_score, on_step=True, on_epoch=True)
        self.log("f1", self.f1_score, on_step=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):  # type: ignore
        raise NotImplementedError

    @property
    def learning_rate(self) -> float:
        """
        Just a learning rate
        """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, new_lr: float) -> None:
        """
        Avoid getting different on config and object

        Useful if we use the lr finder from
        pytorch lightning

        Parameters
        ----------
        x : float
            [description]
        """
        self.lit_args.learning_rate = new_lr
        self._learning_rate = new_lr
