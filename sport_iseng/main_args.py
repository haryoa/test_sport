"""
Sports iseng module
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base_config import BaseConfig  # type: ignore


@dataclass
class MainDataArgs(BaseConfig):  # type: ignore
    """
    Main Data ARGS

    Parameters
    ----------
    data_csv: str
        CSV contains list of file image
    base_dir: str
        Base directory of list path in the csv file
    extracted: List[str]
        Kind of tpe of data in csv that want to be extracted
        (usually train valid test)
    """

    data_csv: str
    base_dir: str
    extracted: List[str] = field(default_factory=list)


@dataclass
class DataConfigArgs(BaseConfig):  # type: ignore
    """
    Data Config for DataLitModule

    Parameters
    ----------
    train_batch_size: int
        Training batch size
    dev_batch_size: int
        Development or validation batch size
    """

    train_batch_size: int
    dev_batch_size: int


@dataclass
class ModelingArgs(BaseConfig):  # type: ignore
    """
    Modeling ARGS for model

    Parameters
    ----------
    early_stopping_cfg: Dict[str, Any]
        Early stopping config in Pytorch LIghtning
    model_checkpoint_cfg: Dict[str, Any]
        Model heckkpoint in pylit
    train_cfg: Dict[str, Any]
        Trainer cfg in pylit
    model_cfg: Dict[str, Any]
        Model cfg in pylit
    wandb_args: wandb: Optional[Dict[str, Any]]
        wandb args for logger
    """

    early_stopping_cfg: Dict[str, Any] = field(default_factory=dict)
    model_checkpoint_cfg: Dict[str, Any] = field(default_factory=dict)
    trainer_cfg: Dict[str, Any] = field(default_factory=dict)
    model_cfg: Dict[str, Any] = field(default_factory=dict)
    wandb: Optional[Dict[str, Any]] = None
