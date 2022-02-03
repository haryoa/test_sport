"""
Lit data module for pytorch lightning module
"""

from typing import Callable, List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .image_dataset import ImageDataset


class LitImageDataModule(  # pylint: disable=too-many-instance-attributes
    LightningDataModule
):
    """

    Lightning Image for training purpose
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        train_imgs: List[str],
        dev_imgs: List[str],
        train_labels: List[str],
        dev_labels: List[str],
        processer: Callable[[str], torch.Tensor],
        train_batch_size: int = 4,
        dev_batch_size: int = 16,
    ):
        """
        Instantiate lit module

        Parameters
        ----------
        train_imgs : List[str]
            Train image path file
        dev_imgs : List[str]
            Dev image path file
        train_labels : List[str]
            Train label for each instance
        dev_labels : List[str]
            Dev label for each instance
        train_batch_size : int, optional
            batch size for training, by default 4
        dev_batch_size : int, optional
            batch size for dev, by default 16
        processer: Callable[[str], torch.Tensor]
            Data input processer from read image to its tensor
        """
        super().__init__()  # type: ignore
        self.train_imgs = train_imgs
        self.dev_imgs = dev_imgs
        self.train_batch_size = train_batch_size
        self.train_labels = train_labels
        self.dev_labels = dev_labels
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.train_dataset: Optional[ImageDataset] = None
        self.dev_dataset: Optional[ImageDataset] = None
        self.processer = processer

    def get_num_labels(self) -> int:
        """
        Get number of labels according to train dataset

        Returns
        -------
        int
            return number of label, -1 if not instantiated
        """

        num_label = (
            len(self.train_dataset.itos_label)
            if self.train_dataset is not None
            else len(set(self.train_labels))
        )
        return num_label

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup for fit only
        """
        self.train_dataset = ImageDataset(
            self.train_imgs, labels=self.train_labels, processer=self.processer
        )
        self.dev_dataset = ImageDataset(
            self.dev_imgs, labels=self.dev_labels, processer=self.processer
        )

    @classmethod
    def produce_dataloader(
        cls, dataset: ImageDataset, batch_size: int, mode: str, num_workers: int = 0
    ) -> DataLoader[torch.Tensor]:
        """Produce dataloader that can be used even without instantiating obj

        Parameters
        ----------
        dataset : ImageDataset
            data
        batch_size : int
            batch size
        mode : str
            choice "eval" or "train"
        num_workers: int
            number of worker for this data

        Returns
        -------
        DataLoader
            return the created dataloader
        """
        bts = batch_size
        shuffle = mode == "train"
        data = DataLoader(dataset, bts, shuffle=shuffle, num_workers=num_workers)  # type: ignore
        return data

    def train_dataloader(self) -> DataLoader[torch.Tensor]:
        """
        Train Dataloader
        """
        if self.train_dataset is not None:  # pylint: disable=no-else-return
            train_dataset = ImageDataset(
                self.train_imgs, labels=self.train_labels, processer=self.processer
            )
            return self.produce_dataloader(
                train_dataset, self.train_batch_size, mode="train"
            )
        else:
            raise Exception("must be setup first")

    def val_dataloader(self) -> DataLoader[torch.Tensor]:
        """
        Validation dataloader
        """
        if self.dev_dataset is not None:  # pylint: disable=no-else-return
            dev_dataset = ImageDataset(
                self.dev_imgs, labels=self.dev_labels, processer=self.processer
            )
            return self.produce_dataloader(
                dev_dataset, self.train_batch_size, mode="eval"
            )
        else:
            raise Exception("must be setup first")

    def predict_dataloader(self) -> DataLoader[torch.Tensor]:
        """
        predict Dataloader
        """
        raise Exception

    def test_dataloader(self) -> DataLoader[torch.Tensor]:
        """
        test Dataloader
        """
        raise Exception
