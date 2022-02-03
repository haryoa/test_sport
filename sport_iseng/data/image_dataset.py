"""
Image dataset for training a DL
"""

from typing import Callable, Dict, List, Union, Optional
import torch


class ImageDataset:
    """
    Image Dataset for training an image classification
    """

    def __init__(
        self,
        image_paths: List[str],
        processer: Callable[[str], torch.Tensor],
        labels: Optional[List[str]] = None,
        itos_label_set: Optional[List[str]] = None,
    ) -> None:
        """
        Image dataset

        Parameters
        ----------
        image_paths : List[str]
            Path of images that will be read
        labels : List[str]
            Label of the images
        processer : Callable[[str], torch.Tensor]
            Image processer to read and transform it
        """
        self.image_paths = image_paths
        self.labels = labels
        if self.labels is not None:
            self.itos_label = (
                sorted(list(set(self.labels)))
                if itos_label_set is None
                else itos_label_set
            )
            sorted(self.itos_label)
            self.stoi_label = {j: i for i, j in enumerate(self.itos_label)}
        self.processer = processer

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        image_tensor = self.processer(self.image_paths[idx])
        dict_returned: Dict[str, Union[torch.Tensor, int]] = {"image": image_tensor}
        if self.labels is not None:
            dict_returned["label"] = self.stoi_label[self.labels[idx]]
        return dict_returned
