"""
Module to process some images
"""
from typing import Any, Callable, Union
from numpy.typing import NDArray
import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def read_image(path: str) -> NDArray[Any]:
    """
    Read Image using opencv2

    Parameters
    ----------
    path : str
        The path file of the image

    Returns
    -------
    np.ndarray
        Return the image matrix RGB
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img  # type: ignore


def get_data_processer_effnet(read_file: bool = False) -> Callable[[Any], torch.Tensor]:
    """
    Get data processer for efficient net

    Parameters
    ----------
    read_file : bool, optional
        Option to read from file or not.
        If not image should be read from the file
        , by default False

    Returns
    -------
    Callable[[Any], torch.Tensor]
        Return a function that can process an input
    """

    def returned_func(inp: Union[str, NDArray[Any]]) -> torch.Tensor:
        if read_file and isinstance(inp, str):
            inp = read_image(inp)

        tfms = A.Compose(
            [
                A.Resize(  # pylint: disable=no-value-for-parameter
                    height=224, width=224
                ),
                A.Normalize(  # pylint: disable=no-value-for-parameter
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(transpose_mask=True),
            ]
        )
        return tfms(image=inp)["image"]  # type: ignore

    return returned_func
