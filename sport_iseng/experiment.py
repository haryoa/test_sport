"""
Experiment runner goes here
"""
import logging
from typing import List, Optional

import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from rich.logging import RichHandler

from sport_iseng.util import get_image_paths_with_labels

from .data import LitImageDataModule, get_data_processer_effnet
from .main_args import DataConfigArgs, MainDataArgs, ModelingArgs
from .train import LitImageArgs, LitImageClassifier

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")


class ExperimentRunner:
    """
    Runner Experiment
    """

    def __init__(
        self,
        cfg: str,
        test_run: bool,
        gpus: Optional[List[int]] = None,
        ckpt_path: Optional[str] = None,
    ) -> None:
        """
        Initialization of experiment runner

        Parameters
        ----------
        cfg : str
            Yaml config file path
        gpus : Optional[List[str]]
            Gpus used
        test_run : bool
            Whether the run is test or not
        ckpt_path: Optional[str]
            Checkpoint path to resume
        """
        self.cfg = cfg
        self.gpus = gpus
        self.test_run = test_run
        self.ckpt_path = ckpt_path
        self._construct_cfg()

    def _construct_cfg(self) -> None:
        """
        Config constructor from a yaml file
        """
        cfg_loaded = yaml.safe_load(open(self.cfg, "r+"))
        self.main_data_args = MainDataArgs(**cfg_loaded.get("data"))
        self.modeling_args = ModelingArgs(**cfg_loaded.get("modeling"))
        self.data_config_args = DataConfigArgs(**cfg_loaded.get("data_config"))

    def run_experiment(self) -> None:
        """
        Experiment pipeline goes here!
        """
        log.info("[red] EXPERIMENT [/red] dimulai!")
        dict_extracted = get_image_paths_with_labels(
            csv_file=self.main_data_args.data_csv,
            extracted_dataset=self.main_data_args.extracted,
            base_dir=self.main_data_args.base_dir,
        )
        effnet_data_processer = get_data_processer_effnet(read_file=True)
        lit_datamodule = LitImageDataModule(
            train_imgs=dict_extracted["train"]["image_paths"],
            train_labels=dict_extracted["train"]["labels"],
            dev_imgs=dict_extracted["valid"]["image_paths"],
            dev_labels=dict_extracted["valid"]["image_paths"],
            processer=effnet_data_processer,
            train_batch_size=self.data_config_args.train_batch_size,
            dev_batch_size=self.data_config_args.dev_batch_size,
        )
        callbacks = []
        logger = None
        if not self.test_run:
            early_stopping_clb = EarlyStopping(**self.modeling_args.early_stopping_cfg)
            model_checkpoint_clb = ModelCheckpoint(
                **self.modeling_args.model_checkpoint_cfg
            )
            callbacks.extend([early_stopping_clb, model_checkpoint_clb])
            if self.modeling_args.wandb is not None:
                logger = WandbLogger(**self.modeling_args.wandb)
        lit_trainer = Trainer(
            gpus=self.gpus,
            fast_dev_run=self.test_run,
            callbacks=callbacks,
            logger=logger,
            **self.modeling_args.trainer_cfg
        )

        model_args = LitImageArgs(
            num_labels=lit_datamodule.get_num_labels(), **self.modeling_args.model_cfg
        )
        model = LitImageClassifier(model_args)
        lit_trainer.fit(model, datamodule=lit_datamodule, ckpt_path=self.ckpt_path)
